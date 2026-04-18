"""
AnatomyAtlas.py — Interactive 3D Anatomy Atlas for 3D Slicer.

Features:
  - Loads a multi-label segmentation (from TotalSegmentator or demo phantom)
  - Colours each organ by anatomical system
  - Click any structure in the 3D view to see its name, volume, and clinical notes
  - System toggle buttons (show/hide organ groups)
  - Quiz mode: structures highlighted anonymously, student types the name

Install in Slicer:
  Edit → Application Settings → Modules → Additional module paths
  → add: /path/to/anatomy-atlas/AnatomyAtlas
  Restart Slicer → find "Anatomy Atlas" under Education category
"""
import json
import os
import random
import sys
import slicer
import vtk
import numpy as np
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)
from slicer.util import VTKObservationMixin

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "Resources", "anatomy_data.json")

# Maximum distance (mm) from a segment surface to count as a hit
_PICK_THRESHOLD_MM = 8.0


# ---------------------------------------------------------------------------
# Module descriptor
# ---------------------------------------------------------------------------

class AnatomyAtlas(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title        = "Anatomy Atlas"
        parent.categories   = ["Education"]
        parent.dependencies = []
        parent.contributors = ["Ke Wang (Medical Imaging Engineer)"]
        parent.helpText     = (
            "Interactive 3D Anatomy Atlas. Load a segmentation, explore organs "
            "by system, and test your knowledge with the built-in quiz."
        )
        parent.acknowledgementText = "Part of the Medical Imaging Engineering Portfolio — kwang316.github.io"


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class AnatomyAtlasWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic          = None
        self._quiz_mode     = False
        self._quiz_organ    = None
        self._quiz_score    = {"correct": 0, "total": 0}
        self._active_system = "all"
        self._press_pos     = None   # used to distinguish click from drag

    def setup(self):
        super().setup()
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/AnatomyAtlas.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        self.logic = AnatomyAtlasLogic(_DATA)

        # Wire system buttons
        self.ui.btnAll.connect("clicked()", lambda: self._filter_system("all"))
        self.ui.btnDigestive.connect("clicked()", lambda: self._filter_system("digestive"))
        self.ui.btnVascular.connect("clicked()", lambda: self._filter_system("vascular"))
        self.ui.btnRespiratory.connect("clicked()", lambda: self._filter_system("respiratory"))
        self.ui.btnUrinary.connect("clicked()", lambda: self._filter_system("urinary"))
        self.ui.btnSkeletal.connect("clicked()", lambda: self._filter_system("skeletal"))
        self.ui.btnLymphatic.connect("clicked()", lambda: self._filter_system("lymphatic"))

        self.ui.organList.connect("itemClicked(QListWidgetItem*)", self._on_list_click)
        self.ui.quizButton.connect("clicked()", self._toggle_quiz)
        self.ui.quizSubmit.connect("clicked()", self._check_quiz_answer)
        self.ui.quizAnswer.connect("returnPressed()", self._check_quiz_answer)

        self._populate_organ_list("all")
        self._setup_3d_click_observer()

        # Auto-apply colours if a segmentation is already in the scene when the
        # module opens, and observe future NodeAdded events.
        self._observe_scene()
        seg_node = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
        if seg_node:
            self.logic.load_segmentation(seg_node)
            self.ui.statusLabel.setText(f"Loaded: {seg_node.GetName()}")

    # ------------------------------------------------------------------
    # Scene observation — auto-applies colours when a segmentation lands
    # ------------------------------------------------------------------

    def _observe_scene(self):
        self.addObserver(
            slicer.mrmlScene,
            slicer.mrmlScene.NodeAddedEvent,
            self._on_node_added,
        )

    @vtk.calldata_type(vtk.VTK_OBJECT)
    def _on_node_added(self, caller, event, call_data):
        if isinstance(call_data, slicer.vtkMRMLSegmentationNode):
            self.logic.load_segmentation(call_data)
            self.ui.statusLabel.setText(f"Loaded: {call_data.GetName()}")

    # ------------------------------------------------------------------
    # Organ list
    # ------------------------------------------------------------------

    def _populate_organ_list(self, system_filter: str):
        import qt
        self.ui.organList.clear()
        for key, data in self.logic.organs.items():
            if system_filter != "all" and data.get("system") != system_filter:
                continue
            color = data.get("color", [0.7, 0.7, 0.7])
            item  = qt.QListWidgetItem(f"  {data['display_name']}")
            px    = qt.QPixmap(14, 14)
            px.fill(qt.QColor(
                int(color[0] * 255),
                int(color[1] * 255),
                int(color[2] * 255),
            ))
            item.setIcon(qt.QIcon(px))
            item.setData(qt.Qt.UserRole, key)
            self.ui.organList.addItem(item)

    def _filter_system(self, system: str):
        self._active_system = system
        self._populate_organ_list(system)
        self._apply_visibility(system)
        sys_info = self.logic.systems.get(system, {})
        self.ui.statusLabel.setText(
            f"Showing: {sys_info.get('label', 'All systems')}"
            if system != "all" else "Showing: all systems"
        )

    # ------------------------------------------------------------------
    # Info panel
    # ------------------------------------------------------------------

    def _show_organ_info(self, organ_key: str, computed_volume_ml: float = None):
        if organ_key not in self.logic.organs:
            return
        data = self.logic.organs[organ_key]

        self.ui.organNameLabel.setText(data["display_name"])
        sys_info = self.logic.systems.get(data.get("system", ""), {})
        self.ui.systemBadge.setText(
            f"{sys_info.get('icon','')}  {sys_info.get('label','')}"
        )

        if computed_volume_ml:
            self.ui.volValue.setText(f"{computed_volume_ml:.0f} mL")
        else:
            self.ui.volValue.setText("—")

        vr = data.get("volume_range_ml")
        self.ui.rangeValue.setText(f"{vr[0]}–{vr[1]} mL" if vr else "—")
        self.ui.clinicalNotes.setPlainText(data.get("clinical_notes", ""))

        fact = data.get("fun_fact", "")
        self.ui.funFactLabel.setText(f"💡 {fact}" if fact else "")

        self._highlight_organ(organ_key)

    # ------------------------------------------------------------------
    # 3D view click observer
    # Bug fixes:
    #   1. Track press position so camera drags don't trigger an identify.
    #   2. Use RAS-to-segment surface distance (vtkImplicitPolyDataDistance)
    #      instead of the never-populated actor_to_organ dict.
    # ------------------------------------------------------------------

    def _setup_3d_click_observer(self):
        interactor = (
            slicer.app.layoutManager()
            .threeDWidget(0)
            .threeDView()
            .renderWindow()
            .GetInteractor()
        )
        self._press_tag  = interactor.AddObserver("LeftButtonPressEvent",   self._on_3d_press)
        self._click_tag  = interactor.AddObserver("LeftButtonReleaseEvent", self._on_3d_click)

    def _on_3d_press(self, caller, event):
        self._press_pos = caller.GetEventPosition()

    def _on_3d_click(self, caller, event):
        release_pos = caller.GetEventPosition()
        if self._press_pos is not None:
            dx = abs(release_pos[0] - self._press_pos[0])
            dy = abs(release_pos[1] - self._press_pos[1])
            if dx > 5 or dy > 5:
                return  # camera drag — ignore

        x, y = release_pos
        renderer = (
            slicer.app.layoutManager()
            .threeDWidget(0)
            .threeDView()
            .renderWindow()
            .GetRenderers()
            .GetFirstRenderer()
        )
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        picker.Pick(x, y, 0, renderer)

        if picker.GetCellId() == -1:
            return

        pick_pos = picker.GetPickPosition()
        seg_node = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
        if not seg_node:
            return

        organ_key = self.logic.find_organ_at_ras(pick_pos, seg_node)
        if not organ_key:
            return

        if self._quiz_mode:
            self._quiz_organ = organ_key
            self.ui.organNameLabel.setText("❓ Name this structure")
            self.ui.systemBadge.setText("")
            self.ui.clinicalNotes.setPlainText("")
            self.ui.funFactLabel.setText("")
            self.ui.quizAnswer.setFocus()
            self._highlight_organ(organ_key)
            return

        vol = self.logic.compute_volume_ml(organ_key)
        self._show_organ_info(organ_key, vol)

        import qt
        for i in range(self.ui.organList.count()):
            item = self.ui.organList.item(i)
            if item.data(qt.Qt.UserRole) == organ_key:
                self.ui.organList.setCurrentItem(item)
                break

    def _on_list_click(self, item):
        import qt
        key = item.data(qt.Qt.UserRole)
        vol = self.logic.compute_volume_ml(key)
        self._show_organ_info(key, vol)

    # ------------------------------------------------------------------
    # 3D visibility toggle
    # ------------------------------------------------------------------

    def _apply_visibility(self, system_filter: str):
        seg_node = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
        if not seg_node:
            return
        seg = seg_node.GetSegmentation()
        for key, data in self.logic.organs.items():
            seg_id = seg.GetSegmentIdBySegmentName(data["display_name"])
            if not seg_id:
                continue
            visible = (system_filter == "all" or data.get("system") == system_filter)
            seg_node.GetDisplayNode().SetSegmentVisibility(seg_id, visible)
        slicer.util.forceRenderAllWindows()

    # ------------------------------------------------------------------
    # Organ highlight
    # ------------------------------------------------------------------

    def _highlight_organ(self, organ_key: str):
        seg_node = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
        if not seg_node:
            return
        seg = seg_node.GetSegmentation()
        target_name = self.logic.organs[organ_key]["display_name"]
        for i in range(seg.GetNumberOfSegments()):
            sid  = seg.GetNthSegmentID(i)
            name = seg.GetSegment(sid).GetName()
            seg_node.GetDisplayNode().SetSegmentOpacity3D(
                sid, 1.0 if name == target_name else 0.15
            )
        slicer.util.forceRenderAllWindows()

    # ------------------------------------------------------------------
    # Quiz mode
    # Bug fix: matching now requires the full organ name or its longest
    # distinctive word (≥4 chars) to appear in the answer.
    # ------------------------------------------------------------------

    def _toggle_quiz(self):
        self._quiz_mode = not self._quiz_mode
        if self._quiz_mode:
            self._quiz_score = {"correct": 0, "total": 0}
            self.ui.quizButton.setText("🎯  Stop Quiz")
            self.ui.quizWidget.setVisible(True)
            self.ui.statusLabel.setText("QUIZ MODE — click a structure to identify it")
            self._highlight_random_organ()
        else:
            self.ui.quizButton.setText("🎯  Start Quiz")
            self.ui.quizWidget.setVisible(False)
            self.ui.quizResult.setText("")
            self.ui.statusLabel.setText("Quiz ended")
            self._reset_opacity()

    def _highlight_random_organ(self):
        visible_organs = list(self.logic.organs.keys())
        if self._active_system != "all":
            visible_organs = [k for k, v in self.logic.organs.items()
                              if v.get("system") == self._active_system]
        if not visible_organs:
            return
        self._quiz_organ = random.choice(visible_organs)
        self._highlight_organ(self._quiz_organ)
        self.ui.organNameLabel.setText("❓ Name the highlighted structure")
        self.ui.quizAnswer.clear()

    def _check_quiz_answer(self):
        if not self._quiz_organ:
            return
        answer  = self.ui.quizAnswer.text().strip().lower()
        correct = self.logic.organs[self._quiz_organ]["display_name"].lower()
        is_right = _quiz_match(answer, correct)

        self._quiz_score["total"] += 1
        if is_right:
            self._quiz_score["correct"] += 1
            self.ui.quizResult.setText(
                f"✅  Correct! — {self.logic.organs[self._quiz_organ]['display_name']}  "
                f"({self._quiz_score['correct']}/{self._quiz_score['total']})"
            )
            self.ui.quizResult.setStyleSheet("color: #22c55e; font-size: 13px; font-weight: bold;")
        else:
            self.ui.quizResult.setText(
                f"❌  It was: {self.logic.organs[self._quiz_organ]['display_name']}  "
                f"({self._quiz_score['correct']}/{self._quiz_score['total']})"
            )
            self.ui.quizResult.setStyleSheet("color: #ef4444; font-size: 13px; font-weight: bold;")

        vol = self.logic.compute_volume_ml(self._quiz_organ)
        self._show_organ_info(self._quiz_organ, vol)
        slicer.app.processEvents()
        import qt
        qt.QTimer.singleShot(2500, self._highlight_random_organ)

    def _reset_opacity(self):
        seg_node = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
        if not seg_node:
            return
        seg = seg_node.GetSegmentation()
        for i in range(seg.GetNumberOfSegments()):
            sid = seg.GetNthSegmentID(i)
            seg_node.GetDisplayNode().SetSegmentOpacity3D(sid, 1.0)
        slicer.util.forceRenderAllWindows()

    def cleanup(self):
        try:
            interactor = (
                slicer.app.layoutManager()
                .threeDWidget(0)
                .threeDView()
                .renderWindow()
                .GetInteractor()
            )
            interactor.RemoveObserver(self._press_tag)
            interactor.RemoveObserver(self._click_tag)
        except Exception:
            pass
        self.removeObservers()


# ---------------------------------------------------------------------------
# Quiz matching helper
# ---------------------------------------------------------------------------

def _quiz_match(answer: str, correct: str) -> bool:
    """
    Accept the answer if:
      - it equals the correct name exactly, OR
      - it contains the full correct name, OR
      - it contains the most distinctive word (longest, ≥4 chars) of the
        correct name — e.g. "kidney" matches "Right Kidney".
    Single short words like "right" or "left" are deliberately excluded so
    students can't guess directional words and accidentally pass.
    """
    if not answer:
        return False
    answer  = answer.strip().lower()
    correct = correct.strip().lower()
    if answer == correct or correct in answer:
        return True
    words = sorted(correct.split(), key=len, reverse=True)
    distinctive = next((w for w in words if len(w) >= 4 and w not in ("left", "right", "upper", "lower")), None)
    return bool(distinctive and distinctive in answer)


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------

class AnatomyAtlasLogic(ScriptedLoadableModuleLogic):
    def __init__(self, data_path: str):
        super().__init__()
        with open(data_path) as f:
            db = json.load(f)
        self.organs  = db["organs"]
        self.systems = db["systems"]

    def load_segmentation(self, seg_node) -> None:
        """
        Apply colours from anatomy_data to each segment by display name.
        Must be called once after a segmentation node is loaded into the scene.
        Previously this was never called, so colours were never applied.
        """
        seg     = seg_node.GetSegmentation()
        display = seg_node.GetDisplayNode()

        for key, data in self.organs.items():
            name   = data["display_name"]
            seg_id = seg.GetSegmentIdBySegmentName(name)
            if not seg_id:
                continue
            color = data["color"]
            seg.GetSegment(seg_id).SetColor(*color)

        display.SetVisibility3D(True)
        display.SetVisibility2D(True)
        slicer.util.forceRenderAllWindows()

    def find_organ_at_ras(self, ras_pos, seg_node) -> str | None:
        """
        Return the organ key whose closed-surface representation is closest
        to ras_pos (in RAS mm), or None if nothing is within _PICK_THRESHOLD_MM.

        Replaces the old actor_to_organ dict which was never populated because
        VTK actors for individual segments are not directly accessible via the
        Slicer segmentation display node API.
        """
        seg = seg_node.GetSegmentation()

        # Ensure closed-surface representations exist
        if not seg.ContainsRepresentation("Closed surface"):
            slicer.modules.segmentations.logic().CreateSurfaceRepresentation(seg_node)

        min_dist = float("inf")
        best_key = None

        for key, data in self.organs.items():
            seg_id = seg.GetSegmentIdBySegmentName(data["display_name"])
            if not seg_id:
                continue
            surface = seg.GetSegment(seg_id).GetRepresentation("Closed surface")
            if not surface or surface.GetNumberOfPoints() == 0:
                continue
            imp = vtk.vtkImplicitPolyDataDistance()
            imp.SetInput(surface)
            dist = abs(imp.EvaluateFunction(list(ras_pos)))
            if dist < min_dist:
                min_dist = dist
                best_key = key

        return best_key if min_dist <= _PICK_THRESHOLD_MM else None

    def compute_volume_ml(self, organ_key: str) -> float | None:
        seg_node = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
        if not seg_node or organ_key not in self.organs:
            return None
        seg  = seg_node.GetSegmentation()
        name = self.organs[organ_key]["display_name"]
        sid  = seg.GetSegmentIdBySegmentName(name)
        if not sid:
            return None
        stats = slicer.modules.segmentstatistics.logic()
        stats.SetMRMLScene(slicer.mrmlScene)
        stats.GetParameterNode().SetParameter("Segmentation", seg_node.GetID())
        stats.ComputeStatistics()
        results = stats.GetStatistics()
        vol_cc  = results.GetValueForSegmentAndPlugin(sid, "LabelmapSegmentStatisticsPlugin", "Volume mm3")
        return vol_cc / 1000.0 if vol_cc else None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class AnatomyAtlasTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.delayDisplay("Starting AnatomyAtlas tests")
        self._test_data_loads()
        self._test_all_organs_have_required_keys()
        self._test_quiz_matching()
        self.delayDisplay("AnatomyAtlas tests PASSED")

    def _test_data_loads(self):
        logic = AnatomyAtlasLogic(_DATA)
        assert len(logic.organs) > 0, "No organs loaded from anatomy_data.json"
        assert len(logic.systems) > 0, "No systems loaded from anatomy_data.json"
        self.delayDisplay("Data load test PASSED")

    def _test_all_organs_have_required_keys(self):
        logic    = AnatomyAtlasLogic(_DATA)
        required = {"label", "display_name", "system", "color", "clinical_notes"}
        for key, data in logic.organs.items():
            missing = required - set(data.keys())
            assert not missing, f"{key} missing keys: {missing}"
        self.delayDisplay("Schema validation test PASSED")

    def _test_quiz_matching(self):
        assert _quiz_match("liver", "Liver")
        assert _quiz_match("right kidney", "Right Kidney")
        assert _quiz_match("kidney", "Right Kidney")        # distinctive word
        assert not _quiz_match("right", "Right Kidney")     # directional word excluded
        assert not _quiz_match("left", "Left Lung")         # directional word excluded
        assert _quiz_match("lung", "Left Lung")             # distinctive word
        assert _quiz_match("aorta", "Aorta")
        assert _quiz_match("vertebral", "Vertebral Column")
        assert not _quiz_match("", "Liver")
        self.delayDisplay("Quiz matching test PASSED")
