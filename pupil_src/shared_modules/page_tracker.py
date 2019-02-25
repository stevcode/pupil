"""
  Pupil Player Third Party Plugins by cpicanco
  Copyright (C) 2016-2017 Rafael Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
# modified version of marker_detector

import sys, os, platform
import cv2
import numpy as np

from file_methods import Persistent_Dict, load_object
from pyglui.cygl.utils import draw_points, draw_polyline, RGBA
from pyglui import ui
from OpenGL.GL import GL_POLYGON
from methods import normalize, denormalize
from glfw import *
from plugin import Plugin

from square_marker_detect import (detect_markers, detect_markers_robust, draw_markers, m_marker_to_screen, )
from reference_surface import Reference_Surface

from page_detector import detect_pages

from math import sqrt

# logging
import logging

logger = logging.getLogger(__name__)


class Page_Tracker(Plugin):
    icon_chr = chr(0xEC07)
    icon_font = "pupil_icons"

    def __init__(self, g_pool, mode="Show Markers and Surfaces", min_marker_perimeter=60, invert_image=False,
            robust_detection=True, ):
        super().__init__(g_pool)
        self.order = 0.2

        # all markers that are detected in the most recent frame
        self.markers = []

        # self.load_surface_definitions_from_file()
        self.surface_definitions = []
        self.surfaces = []

        # edit surfaces
        self.edit_surfaces = []
        self.edit_surf_verts = []
        self.marker_edit_surface = None
        # plugin state
        self.mode = mode
        self.running = True

        self.robust_detection = robust_detection
        self.aperture = 11
        self.min_marker_perimeter = min_marker_perimeter
        self.min_id_confidence = 0.0
        self.locate_3d = False
        self.invert_image = invert_image

        self.img_shape = None
        self._last_mouse_pos = 0, 0

        self.menu = None
        self.button = None
        self.add_button = None

    def add_surface(self, _):
        surf = Reference_Surface(self.g_pool)
        # surf.on_finish_define = self.save_surface_definitions_to_file
        self.surfaces.append(surf)
        self.update_gui_markers()

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Page Tracker"

        self.button = ui.Thumb("running", self, label="P", hotkey="p")
        self.button.on_color[:] = (0.1, 0.2, 1.0, 0.8)
        self.g_pool.quickbar.append(self.button)
        self.add_button = ui.Thumb("add_surface", setter=self.add_surface, getter=lambda: False, label="A", hotkey="a")
        self.g_pool.quickbar.append(self.add_button)
        self.update_gui_markers()

    def deinit_ui(self):
        self.g_pool.quickbar.remove(self.button)
        self.button = None
        self.g_pool.quickbar.remove(self.add_button)
        self.add_button = None
        self.remove_menu()

    def update_gui_markers(self):
        self.menu.elements[:] = []
        self.menu.append(ui.Info_Text(
            "This plugin detects a page in the scene. Ideally the page is a white rectangle on a black backgroud. Place 1 page in the world view and click *Add surface*."))
        self.menu.append(ui.Switch("robust_detection", self, label='Robust detection'))
        self.menu.append(ui.Slider("min_marker_perimeter", self, step=1, min=10, max=500))
        self.menu.append(ui.Switch("locate_3d", self, label='3D localization'))
        self.menu.append(
            ui.Selector("mode", self, label="Mode", selection=["Show Markers and Surfaces", "Show marker IDs"]))
        self.menu.append(ui.Button("Add surface", lambda: self.add_surface("_"), ))

        for s in self.surfaces:
            idx = self.surfaces.index(s)
            s_menu = ui.Growing_Menu("Page %s" % idx)
            s_menu.collapsed = True
            s_menu.append(ui.Text_Input("name", s))
            s_menu.append(ui.Text_Input("x", s.real_world_size, label='X size'))
            s_menu.append(ui.Text_Input("y", s.real_world_size, label='Y size'))
            s_menu.append(ui.Button("Open Debug Window", s.open_close_window))

            # closure to encapsulate idx
            def make_remove_s(i):
                return lambda: self.remove_surface(i)

            remove_s = make_remove_s(idx)
            s_menu.append(ui.Button("remove", remove_s))
            self.menu.append(s_menu)

    def recent_events(self, events):
        frame = events.get("frame")
        if not frame:
            return
        self.img_shape = frame.height, frame.width, 3

        if self.running:
            gray = frame.gray
            if self.invert_image:
                gray = 255 - gray

            # hack "self.markers" instead "self.screens" is kept for inheritence compatibility
            self.markers = detect_pages(gray)

            if self.mode == "Show marker IDs":
                draw_markers(frame.img, self.markers)
                # events['frame'] = frame        Not in original, delete if this works

        # locate surfaces, map gaze
        for s in self.surfaces:
            s.locate(self.markers, self.min_marker_perimeter, self.min_id_confidence, self.locate_3d, )
            if s.detected:
                s.gaze_on_srf = s.map_data_to_surface(events.get("gaze", []), s.m_from_screen)
                s.fixations_on_srf = s.map_data_to_surface(events.get("fixations", []), s.m_from_screen)
                s.update_gaze_history()
            else:
                s.gaze_on_srf = []
                s.fixations_on_srf = []

        events["surfaces"] = []
        for s in self.surfaces:
            if s.detected:
                datum = {
                    "topic": "surfaces.{}".format(s.name),
                    "name": s.name,
                    "uid": s.uid,
                    "m_to_screen": s.m_to_screen.tolist(),
                    "m_from_screen": s.m_from_screen.tolist(),
                    "gaze_on_srf": s.gaze_on_srf,
                    "fixations_on_srf": s.fixations_on_srf,
                    "timestamp": frame.timestamp,
                    "camera_pose_3d": s.camera_pose_3d.tolist() if s.camera_pose_3d is not None else None, }
                events["surfaces"].append(datum)

        if self.running:
            self.button.status_text = "{}/{}".format(len([s for s in self.surfaces if s.detected]), len(self.surfaces))
        else:
            self.button.status_text = "tracking paused"

        if self.mode == "Show Markers and Surfaces":
            # edit surfaces by user
            if self.edit_surf_verts:
                pos = self._last_mouse_pos
                for s, v_idx in self.edit_surf_verts:
                    if s.detected:
                        new_pos = s.img_to_ref_surface(np.array(pos))
                        s.move_vertex(v_idx, new_pos)

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        # self.save_surface_definitions_to_file()

        for s in self.surfaces:
            s.cleanup()
