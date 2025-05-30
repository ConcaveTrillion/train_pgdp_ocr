import pathlib
from enum import Enum
from logging import getLogger

from cv2 import line as cv2_line
from ipywidgets import HTML, Button, HBox, Layout, Text, VBox
from ipywidgets import Image as ipywidgets_Image
from numpy import ndarray  # GridBox,

from pd_book_tools.geometry import BoundingBox
from pd_book_tools.ocr.block import Block
from pd_book_tools.ocr.ground_truth_matching import update_line_with_ground_truth
from pd_book_tools.ocr.image_utilities import (
    get_cropped_encoded_image_scaled_bbox,
    get_cropped_word_image,
    get_encoded_image,
)
from pd_book_tools.ocr.page import Page
from pd_book_tools.ocr.word import Word
from pd_book_tools.pgdp.pgdp_results import PGDPPage
from pd_book_tools.utility.ipynb_widgets import (
    get_formatted_text_html_span,
    get_html_string_from_image_src,
    get_html_widget_from_cropped_image,
)

# Configure logging
logger = getLogger(__name__)
ui_logger = getLogger(__name__ + ".UI")


class EditorTaskType(Enum):
    NONE = 0
    SPLIT = 1
    EDITBBOX = 2


class IpynbLineEditor:
    """
    UI editing a single line
    """

    _current_pgdp_page: PGDPPage
    _current_ocr_page: Page
    _current_ocr_line: Block
    _current_ocr_line_p3_gt_text: str
    line_matches: list[dict]
    page_image_change_callback: callable = None
    line_change_callback: callable = None

    monospace_font_name: str

    GridVBox: VBox
    line_image: ipywidgets_Image
    LineImageHBox: HBox
    OcrLineTextHBox: HBox
    GTLineTextHBox: HBox
    LineActionButtonsHBox: HBox
    WordMatchingTableHBox: HBox
    WordMatchingTableVBoxes: list[VBox]
    TaskHBox: HBox

    task_type: EditorTaskType = EditorTaskType.NONE
    task_match_idx: int = -1
    split_task_x_coordinate: int = -1
    split_task_word_split_idx: int = -1

    basic_box_layout: Layout = Layout(
        margin="0px",
        padding="1px",
        border="1px solid black",
        # flex="1 0",
    )

    red_editor_layout: Layout = Layout(
        margin="0px 0px 5px 5px",
        padding="0px",
        border="3px solid red",
    )
    gray_editor_layout: Layout = Layout(
        margin="0px 0px 5px 5px",
        padding="0px",
        border="3px solid gray",
    )
    green_editor_layout: Layout = Layout(
        margin="0px 0px 5px 5px",
        padding="0px",
        border="3px solid green",
    )

    def init_font(
        self,
        monospace_font_name: str,
        monospace_font_path: pathlib.Path | str,
    ):
        self.monospace_font_name = monospace_font_name
        if isinstance(monospace_font_path, str):
            monospace_font_path = pathlib.Path(monospace_font_path)
        self.monospace_font_path = monospace_font_path

    def __init__(
        self,
        page: Page,
        pgdp_page: PGDPPage,
        line: Block,
        page_image_change_callback: callable = None,
        line_change_callback: callable = None,
        monospace_font_name: str = "Courier New",
        monospace_font_path: pathlib.Path | str = None,
    ):
        self.init_font(
            monospace_font_name=monospace_font_name,
            monospace_font_path=monospace_font_path,
        )

        self.GridVBox = VBox()
        self.GridVBox.children = []
        self.GridVBox.layout = self.gray_editor_layout

        self._current_ocr_page = page
        self._current_pgdp_page = pgdp_page
        self._current_ocr_line = line
        self.line_matches = []
        self.page_image_change_callback = page_image_change_callback
        self.line_change_callback = line_change_callback

        self.redraw_ui()

    def redraw_ui(self):
        ui_logger.debug(f"Redrawing UI for line: {self._current_ocr_line.text[:20]}...")
        self.calculate_line_matches()
        if not self.line_matches:
            ui_logger.debug(
                f"Line {self._current_ocr_line.text} has no matches. Hiding UI."
            )
            # If there are no matches, don't display the UI
            self.GridVBox.children = []
            self.GridVBox.layout = Layout(display="none")
            return
        else:
            ui_logger.debug(f"Line {self._current_ocr_line.text} has matches.")
            if self._current_ocr_line.ground_truth_exact_match:
                self.GridVBox.layout = self.gray_editor_layout
            else:
                self.GridVBox.layout = self.red_editor_layout

        if self._current_ocr_line.additional_block_attributes.get(
            "line_editor_validated"
        ):
            ui_logger.debug(
                f"Line {self._current_ocr_line.text} is marked as validated."
            )
            self.GridVBox.layout = self.green_editor_layout
            ui_logger.debug(f"Line {self._current_ocr_line.text} green border added.")

        ui_logger.debug(
            f"Redrawing components for line: {self._current_ocr_line.text[:20]}..."
        )

        self.draw_ui_fullline_image_hbox()
        self.draw_ui_fullline_ocr_text_hbox()
        self.draw_ui_fullline_gt_text_hbox()
        self.draw_ui_line_actions_hbox()
        self.draw_ui_word_matching_table()
        self.draw_ui_active_task()

        self.rebuild_gridbox_children()

        ui_logger.debug("Line UI redrawn.")

    def rebuild_gridbox_children(self):
        ui_logger.debug(
            f"Rebuilding GridVBox children for line: {self._current_ocr_line.text[:20]}..."
        )
        # Rebuild the GridVBox children
        self.GridVBox.children = []
        self.GridVBox.children = [
            self.LineImageHBox,
            self.OcrLineTextHBox,
            self.GTLineTextHBox,
            self.LineActionButtonsHBox,
            self.WordMatchingTableHBox,
            self.TaskHBox,
        ]
        ui_logger.debug(
            f"GridVBox children rebuilt for line: {self._current_ocr_line.text[:20]}."
        )

    def draw_ui_fullline_image_hbox(self):
        ui_logger.debug("Drawing UI for full line image.")
        self.LineImageHBox = HBox()
        self.LineImageHBox.layout = self.basic_box_layout
        cropped_line_image_html = get_html_widget_from_cropped_image(
            self._current_ocr_page.cv2_numpy_page_image,
            self._current_ocr_line.bounding_box,
        )
        self.LineImageHBox.children = [cropped_line_image_html]

    def draw_ui_fullline_ocr_text_hbox(self):
        ui_logger.debug("Drawing UI for full line OCR text.")
        self.OcrLineTextHBox = HBox()
        self.OcrLineTextHBox.layout = self.basic_box_layout
        if self._current_ocr_line.ground_truth_exact_match:
            linecolor_css = "lightgray"
        else:
            linecolor_css = "unset"

        self.OcrLineTextHBox.children = [
            get_formatted_text_html_span(
                linecolor_css=linecolor_css,
                text=self._current_ocr_line.text,
                font_family_css=self.monospace_font_name,
                font_size_css="14px",
            )
        ]

    def draw_ui_fullline_gt_text_hbox(self):
        ui_logger.debug("Drawing UI for full line GT text.")
        self.GTLineTextHBox = HBox()
        self.GTLineTextHBox.layout = self.basic_box_layout
        if self._current_ocr_line.ground_truth_exact_match:
            linecolor_css = "lightgray"
        else:
            linecolor_css = "unset"

        self.GTLineTextHBox.children = [
            get_formatted_text_html_span(
                linecolor_css=linecolor_css,
                text=(self._current_ocr_line.ground_truth_text or "&nbsp;"),
                font_family_css=self.monospace_font_name,
                font_size_css="14px",
            )
        ]

    def draw_ui_line_actions_hbox(self):
        ui_logger.debug("Drawing UI for line actions.")
        CopyOCRToGTButton = Button(description="Copy Line to GT")
        CopyOCRToGTButton.on_click(self.copy_ocr_to_gt)

        DeleteLineButton = Button(description="Delete Line")
        DeleteLineButton.on_click(self.delete_line)

        MarkValidatedButton = Button(description="Mark as Validated")
        MarkValidatedButton.on_click(self.mark_validated)

        self.LineActionButtonsHBox = HBox()
        self.LineActionButtonsHBox.layout = self.basic_box_layout
        self.LineActionButtonsHBox.children = [
            CopyOCRToGTButton,
            DeleteLineButton,
            MarkValidatedButton,
        ]

    def draw_ui_active_task(self):
        ui_logger.debug("Drawing UI for active task.")
        self.TaskHBox = HBox()
        self.TaskHBox.layout = self.basic_box_layout
        self.TaskHBox.children = []

        if self.task_type == EditorTaskType.SPLIT:
            ui_logger.debug("Split task active.")
            self.draw_ui_split_task()

        elif self.task_type == EditorTaskType.EDITBBOX:
            ui_logger.debug("Edit BBox task active.")
            self.draw_ui_edit_bbox_task()

        return

    def get_split_image_html_widget(self):
        ui_logger.debug("Getting split image.")
        img_ndarray: ndarray = self.line_matches[self.task_match_idx]["img_ndarray"]

        h, w = img_ndarray.shape[:2]

        if self.split_task_x_coordinate == -1:
            # Get the x coordinate of the split
            self.split_task_x_coordinate = int(w / 2)

        # use cv2 to draw a vertical red line on the image based on the split location
        ui_logger.debug("Drawing Line on image.")
        split_img = cv2_line(
            img=img_ndarray.copy(),
            pt1=(self.split_task_x_coordinate, 0),
            pt2=(self.split_task_x_coordinate, h),
            color=(255, 0, 0),
            thickness=1,
        )

        # Encode the split image as PNG and get a <img> tag string
        ui_logger.debug("Getting encoded image.")
        _, _, data_src_string = get_encoded_image(split_img)

        ui_logger.debug("Returning HTML widget.")
        return HTML(
            get_html_string_from_image_src(
                data_src_string=data_src_string, height="height: 36px;"
            )
        )

    def update_split_image(self):
        ui_logger.debug("Updating split image.")
        # Get the split image HTML widget
        split_image_html = self.get_split_image_html_widget()

        ui_logger.debug("Setting up HBox for split image.")
        self.SplitImageHBox.children = [split_image_html]

        ui_logger.debug("Split image updated.")
        return

    def update_split_text(self):
        ui_logger.debug("Updating split ocr text.")

        # Get the split text HTML widget
        ocr_text: str = self.line_matches[self.task_match_idx]["ocr_text"]

        if self.split_task_word_split_idx == -1:
            # half of the word length
            self.split_task_word_split_idx = int(len(ocr_text) / 2)

        # insert | at the split location
        ocr_text = (
            ocr_text[: self.split_task_word_split_idx]
            + "\u00a0"  # non-breaking space
            + ocr_text[self.split_task_word_split_idx :]
        )

        ocr_text = "\u00a0".join(ocr_text) if ocr_text else ""

        splitTextHTML = get_formatted_text_html_span(
            font_family_css=self.monospace_font_name,
            font_size_css="16px",
            text=ocr_text,
        )
        splitTextHTML.layout = Layout(margin="0px 2px 0px 2px")

        ui_logger.debug("Setting up HBox for split image.")
        self.SplitTextHBox.children = [splitTextHTML]

        ui_logger.debug("Split text updated.")
        return

    def split_move_text_callback(self, amount):
        # Move the split line left by <amount> characters
        ui_logger.debug(f"Moving split word by {amount} characters.")
        ui_logger.debug(
            f"Current split character position: {self.split_task_word_split_idx}"
        )
        w = len(self.line_matches[self.task_match_idx]["ocr_text"])
        self.split_task_word_split_idx = int(
            min(max(self.split_task_word_split_idx + amount, 0), w)
        )
        ui_logger.debug(
            f"New split character position: {self.split_task_word_split_idx}"
        )
        self.update_split_text()
        return

    def split_move_pixels_callback(self, amount):
        # Move the split line left by <amount> pixels
        w = self.line_matches[self.task_match_idx]["img_ndarray"].shape[1]
        ui_logger.debug(f"Moving split line by {amount} pixels.")
        ui_logger.debug(f"Current split line position: {self.split_task_x_coordinate}")
        self.split_task_x_coordinate = int(
            min(max(self.split_task_x_coordinate + amount, 0), w)
        )
        ui_logger.debug(f"New split line position: {self.split_task_x_coordinate}")
        self.update_split_image()
        return

    def split_move_percent_callback(self, amount):
        # Move the split line left by <amount> percent
        w = self.line_matches[self.task_match_idx]["img_ndarray"].shape[1]
        ui_logger.debug(f"Moving split line by {amount} percent.")
        ui_logger.debug(f"Current split line position: {self.split_task_x_coordinate}")
        self.split_task_x_coordinate = int(
            min(max(self.split_task_x_coordinate + int(w * amount), 0), w)
        )
        ui_logger.debug(f"New split line position: {self.split_task_x_coordinate}")
        self.update_split_image()
        return

    def draw_ui_split_task(self):
        ui_logger.debug("Drawing UI for split task.")

        self.SplitImageHBox = HBox()
        self.SplitImageHBox.layout = Layout(margin="0px 0px 10px 0px")

        self.update_split_image()

        ui_logger.debug("Generating Buttons for split task.")
        self.CancelSplitButton = Button(
            description="X",
            layout=Layout(width="16px", margin="0px 2px 0px 2px", padding="1px"),
        )
        self.CancelSplitButton.on_click(lambda _: self.cancel_split_task())

        self.SplitArrowLeftButton = Button(
            description="<1p",
            layout=Layout(width="30px", margin="0px 2px 0px 2px", padding="1px"),
        )
        self.SplitArrowLeftButton.on_click(
            lambda _: self.split_move_pixels_callback(-1)
        )

        self.SplitArrowRightButton = Button(
            description=">1p",
            layout=Layout(width="30px", margin="0px 2px 0px 2px", padding="1px"),
        )
        self.SplitArrowRightButton.on_click(
            lambda _: self.split_move_pixels_callback(1)
        )

        self.SplitArrowLeft5Button = Button(
            description="<5%",
            layout=Layout(width="30px", margin="0px 2px 0px 2px", padding="1px"),
        )
        self.SplitArrowLeft5Button.on_click(
            lambda _: self.split_move_percent_callback(-0.05)
        )
        self.SplitArrowRight5Button = Button(
            description=">5%",
            layout=Layout(width="30px", margin="0px 2px 0px 2px", padding="1px"),
        )
        self.SplitArrowRight5Button.on_click(
            lambda _: self.split_move_percent_callback(0.05)
        )
        self.SplitArrowLeft20Button = Button(
            description="<20%",
            layout=Layout(width="40px", margin="0px 2px 0px 2px", padding="1px"),
        )
        self.SplitArrowLeft20Button.on_click(
            lambda _: self.split_move_percent_callback(-0.2)
        )
        self.SplitArrowRight20Button = Button(
            description=">20%",
            layout=Layout(width="40px", margin="0px 2px 0px 2px", padding="1px"),
        )
        self.SplitArrowRight20Button.on_click(
            lambda _: self.split_move_percent_callback(0.2)
        )
        self.SplitOKButton = Button(
            description="Execute Split",
        )
        self.SplitOKButton.on_click(lambda _: self.execute_split())

        self.SplitTextHBox = HBox()
        self.SplitTextHBox.layout = Layout(margin="0px 0px 10px 0px")

        self.update_split_text()

        self.SplitWordArrowLeftButton = Button(
            description="<",
            layout=Layout(width="14px", margin="0px 2px 0px 2px", padding="1px"),
        )
        self.SplitWordArrowLeftButton.on_click(
            lambda _: self.split_move_text_callback(-1)
        )

        self.SplitWordArrowRightButton = Button(
            description=">",
            layout=Layout(width="14px", margin="0px 2px 0px 2px", padding="1px"),
        )
        self.SplitWordArrowRightButton.on_click(
            lambda _: self.split_move_text_callback(1)
        )

        self.SplitVBoxLine1: HBox = HBox()
        self.SplitVBoxLine2: HBox = HBox()

        self.SplitVBoxLine1.children = [
            self.SplitImageHBox,
            self.CancelSplitButton,
            self.SplitArrowLeftButton,
            self.SplitArrowRightButton,
            self.SplitArrowLeft5Button,
            self.SplitArrowRight5Button,
            self.SplitArrowLeft20Button,
            self.SplitArrowRight20Button,
            self.SplitOKButton,
        ]

        self.SplitVBoxLine2.children = [
            self.SplitTextHBox,
            self.SplitWordArrowLeftButton,
            self.SplitWordArrowRightButton,
        ]

        self.SplitVBox: VBox = VBox(layout=Layout(margin="0px 0px 0px 10px"))
        self.SplitVBox.children = [
            self.SplitVBoxLine1,
            self.SplitVBoxLine2,
        ]

        self.TaskHBox.children = [self.SplitVBox]

        ui_logger.debug("Split task UI drawn.")
        return

    def get_edit_bbox(self):
        ui_logger.debug("Getting edit bbox image.")
        img_ndarray: ndarray = self._current_ocr_page.cv2_numpy_page_image

        ui_logger.debug("Getting bounding box.")
        # Get the bounding box of the word
        word_bbox: BoundingBox = self.line_matches[self.task_match_idx][
            "word"
        ].bounding_box

        ui_logger.debug("Scaling bounding box.")
        h, w = img_ndarray.shape[:2]
        copy_bbox_scaled = word_bbox.scale(w, h)

        ui_logger.debug(f"Scaled bounding box: {copy_bbox_scaled.to_ltrb()}")

        # Adjust the bbox with the edit margins
        ui_logger.debug("Adjusting Margins.")

        minX = min(max(copy_bbox_scaled.minX + self.edit_margins[0], 0), w)
        minY = min(max(copy_bbox_scaled.minY + self.edit_margins[1], 0), h)
        maxX = min(max(copy_bbox_scaled.maxX + self.edit_margins[2], 0), w)
        maxY = min(max(copy_bbox_scaled.maxY + self.edit_margins[3], 0), h)

        edit_bbox_scaled = BoundingBox.from_ltrb(minX, minY, maxX, maxY)

        ui_logger.debug("Edit bounding box:")
        ui_logger.debug(f"{edit_bbox_scaled.to_ltrb()}")
        return edit_bbox_scaled

    def get_edit_bbox_image_html_widget(self):
        img_ndarray: ndarray = self._current_ocr_page.cv2_numpy_page_image
        modified_bbox = self.get_edit_bbox()

        ui_logger.debug(
            f"Cropping image to bounding box {str(modified_bbox.to_ltrb())}."
        )
        # Crop the image to the bounding box
        _, _, _, data_src_string = get_cropped_encoded_image_scaled_bbox(
            img=img_ndarray, bounding_box_scaled=modified_bbox
        )

        ui_logger.debug("Returning HTML widget.")
        return HTML(
            get_html_string_from_image_src(
                data_src_string=data_src_string, height="height: 36px;"
            )
        )

    def update_edit_bbox_image(self):
        ui_logger.debug("Updating edit bbox image.")
        # Get the split image HTML widget
        edit_image_html = self.get_edit_bbox_image_html_widget()

        ui_logger.debug("Setting up HBox for split image.")
        self.EditBboxImageHBox.children = [edit_image_html]

        ui_logger.debug("Edit bbox image updated.")
        return

    def edit_bbox_adjust_margin(self, margin: str, amount: int):
        ui_logger.debug(f"Adjusting edit bbox margin: {margin} by {amount}.")

        self.edit_margins = [
            self.edit_margins[0] + (amount if margin == "L" else 0),
            self.edit_margins[1] + (amount if margin == "T" else 0),
            self.edit_margins[2] + (amount if margin == "R" else 0),
            self.edit_margins[3] + (amount if margin == "B" else 0),
        ]

        self.update_edit_bbox_image()
        return

    def edit_bbox_refine(self):
        img_ndarray: ndarray = self._current_ocr_page.cv2_numpy_page_image
        h, w = img_ndarray.shape[:2]

        modified_bbox = self.get_edit_bbox()

        ui_logger.debug("Normalizing bbox.")
        normalized_modified_bbox = modified_bbox.normalize(w, h)
        ui_logger.debug(f"Normalized Edit bbox: {normalized_modified_bbox.to_ltrb()}")

        ui_logger.debug("Refining bbox.")
        normalized_refined_bbox: BoundingBox = normalized_modified_bbox.refine(
            img_ndarray, padding_px=1
        )
        ui_logger.debug(
            f"Refined Edit bounding box: {normalized_refined_bbox.to_ltrb()}"
        )
        refined_bbox = normalized_refined_bbox.scale(w, h)
        ui_logger.debug(f"Scaled Refined Edit bounding box: {refined_bbox.to_ltrb()}")

        ui_logger.debug(f"Edit Margins Before refinement: {self.edit_margins}")

        self.edit_margins = [
            self.edit_margins[0] + int(refined_bbox.minX - modified_bbox.minX),
            self.edit_margins[1] + int(refined_bbox.minY - modified_bbox.minY),
            self.edit_margins[2] + int(refined_bbox.maxX - modified_bbox.maxX),
            self.edit_margins[3] + int(refined_bbox.maxY - modified_bbox.maxY),
        ]
        ui_logger.debug(f"Edit Margins after refinement: {self.edit_margins}")

        self.update_edit_bbox_image()
        return

    def get_ui_edit_bbox_margin_buttons(self, margin: str):
        ui_logger.debug(f"Drawing UI for edit bbox {margin} margin buttons.")

        # margin: 'L' for left, 'T' for top, 'R' for right, 'B' for bottom
        EditBboxMarginText = HTML(
            f"<span style='font-family: monospace; font-size: 16px;'>{margin}:</span>"
        )
        EditBboxMarginAdjustLeft = Button(
            description="←" if margin in ["L", "R"] else "↑",
            layout=Layout(width="14px", margin="0px 2px 0px 2px", padding="1px"),
        )
        EditBboxMarginAdjustLeft.on_click(
            lambda _: self.edit_bbox_adjust_margin(margin, -1)
        )
        EditBboxMarginAdjustRight = Button(
            description="→" if margin in ["L", "R"] else "↓",
            layout=Layout(width="14px", margin="0px 2px 0px 2px", padding="1px"),
        )
        EditBboxMarginAdjustRight.on_click(
            lambda _: self.edit_bbox_adjust_margin(margin, 1)
        )
        return [
            EditBboxMarginText,
            EditBboxMarginAdjustLeft,
            EditBboxMarginAdjustRight,
        ]

    def draw_ui_edit_bbox_task(self):
        self.EditBboxImageHBox = HBox()
        self.EditBboxImageHBox.layout = Layout(margin="0px 0px 10px 0px")

        self.update_edit_bbox_image()

        ui_logger.debug("Generating Buttons for split task.")
        self.CancelEditBBoxButton = Button(
            description="X",
            layout=Layout(width="16px", margin="0px 2px 0px 2px", padding="1px"),
        )
        self.CancelEditBBoxButton.on_click(lambda _: self.cancel_edit_bbox_task())

        self.EditBBoxRefineButton = Button(
            description="Refine",
            layout=Layout(width="100%", margin="0px 2px 0px 2px", padding="1px"),
        )
        self.EditBBoxRefineButton.on_click(lambda _: self.edit_bbox_refine())

        self.EditVBoxLine1 = HBox(
            [self.CancelEditBBoxButton, self.EditBBoxRefineButton]
        )
        self.EditVBoxLine2 = HBox(self.get_ui_edit_bbox_margin_buttons("L"))
        self.EditVBoxLine3 = HBox(self.get_ui_edit_bbox_margin_buttons("T"))
        self.EditVBoxLine4 = HBox(self.get_ui_edit_bbox_margin_buttons("R"))
        self.EditVBoxLine5 = HBox(self.get_ui_edit_bbox_margin_buttons("B"))

        self.EditVBoxSaveButton = Button(
            description="Save",
            layout=Layout(width="100%", margin="0px 2px 0px 2px", padding="1px"),
        )
        self.EditVBoxSaveButton.on_click(lambda _: self.execute_edit_bbox_task())

        self.EditVBoxLine6 = HBox([self.EditVBoxSaveButton])

        self.EditHBox: HBox = HBox()
        self.EditVBox: VBox = VBox(layout=Layout(margin="0px 0px 0px 10px"))

        self.EditHBox.children = [self.EditVBox, self.EditBboxImageHBox]

        self.EditVBox.children = [
            self.EditVBoxLine1,
            self.EditVBoxLine2,
            self.EditVBoxLine3,
            self.EditVBoxLine4,
            self.EditVBoxLine5,
            self.EditVBoxLine6,
        ]

        self.TaskHBox.children = [self.EditHBox]

        ui_logger.debug("Edit BBox task UI drawn.")
        return

    def start_split_task(self, match):
        ui_logger.debug("Starting split task.")
        self.task_type = EditorTaskType.SPLIT
        self.split_task_x_coordinate = -1
        self.split_task_image_width = -1
        self.task_match_idx = match["idx"]
        self.redraw_ui()
        return

    def cancel_split_task(self):
        ui_logger.debug("Canceling split task.")
        self.task_type = EditorTaskType.NONE
        self.task_match_idx = None
        self.split_task_x_coordinate = -1
        self.split_task_image_width = -1
        self.redraw_ui()
        return

    def start_edit_bbox_task(self, match):
        ui_logger.debug("Starting edit bbox task.")
        self.task_type = EditorTaskType.EDITBBOX
        self.task_match_idx = match["idx"]
        self.edit_margins = [0, 0, 0, 0]  # Left, Top, Right, Bottom

        self.redraw_ui()
        return

    def cancel_edit_bbox_task(self):
        ui_logger.debug("Canceling edit bbox task.")
        self.task_type = EditorTaskType.NONE
        self.task_match_idx = None
        self.edit_margins = [0, 0, 0, 0]  # Reset margins

        self.redraw_ui()
        return

    def execute_edit_bbox_task(self):
        ui_logger.debug("Executing edit bbox task.")

        # Get the match
        match = self.line_matches[self.task_match_idx]
        word: Word = match["word"]
        ui_logger.debug(
            f"Executing edit bbox task for match: {match['idx']} | word: {word.text}"
        )
        # Get the bounding box of the word
        word_bbox: BoundingBox = word.bounding_box
        ui_logger.debug(f"Original bounding box: {word_bbox.to_ltrb()}")
        # Adjust the bbox with the edit margins
        h, w = self._current_ocr_page.cv2_numpy_page_image.shape[:2]
        word_bbox = word_bbox.scale(w, h)
        minX = min(max(word_bbox.minX + self.edit_margins[0], 0), w)
        minY = min(max(word_bbox.minY + self.edit_margins[1], 0), h)
        maxX = min(max(word_bbox.maxX + self.edit_margins[2], 0), w)
        maxY = min(max(word_bbox.maxY + self.edit_margins[3], 0), h)
        modified_bbox = BoundingBox.from_ltrb(minX, minY, maxX, maxY)
        ui_logger.debug(f"Modified bounding box: {modified_bbox.to_ltrb()}")
        # normalize the bounding box
        normalized_modified_bbox = modified_bbox.normalize(w, h)
        # Set the bounding box of the word
        word.bounding_box = normalized_modified_bbox
        ui_logger.debug(f"New bounding box set: {word.bounding_box.to_ltrb()}")

        # TODO maybe do this
        # Expand the word bounding boxes if they were shrunk too much or didn't include all the connected pixels
        # ui_logger.debug("Expanding word bounding boxes to content.")
        # word.bounding_box = word.bounding_box.expand_to_content(
        #    image=self._current_ocr_page.cv2_numpy_page_image,
        # )

        # Refine the bounding box to fit the text
        self._current_ocr_line.refine_bounding_boxes(
            image=self._current_ocr_page.cv2_numpy_page_image, padding_px=1
        )
        ui_logger.debug("Word bounding boxes have been refined.")

        self.task_type = EditorTaskType.NONE
        self.task_match_idx = None
        self.edit_margins = [0, 0, 0, 0]  # Reset margins

        self.redraw_ui()

        if self.page_image_change_callback:
            self.page_image_change_callback()

        ui_logger.debug("Edit bbox task executed.")
        return

    def execute_split(self):
        ui_logger.debug("Executing split task.")

        # Get the match
        match = self.line_matches[self.task_match_idx]
        split_word_idx = match["word_idx"]
        ui_logger.debug(
            f"Executing split task for match: {match['idx']} | word index: {split_word_idx}"
        )

        # Convert X coordinate into normalized form
        # convert bbox width to pixels
        w = match["img_ndarray"].shape[1]
        ui_logger.debug(f"word image width: {w}")
        full_w = self._current_ocr_page.cv2_numpy_page_image.shape[1]
        ui_logger.debug(f"full image width: {full_w}")
        ratio = float(w) / float(full_w)
        ui_logger.debug(f"w ratio: {ratio}")
        # convert split x coordinate from pixels to normalized form
        ui_logger.debug(f"Pixel offset: {self.split_task_x_coordinate}")
        normalized_split_x_offset = float(self.split_task_x_coordinate) / float(full_w)
        ui_logger.debug(f"Normalized split offset: {normalized_split_x_offset}")

        # Split the word in the OCR object
        self._current_ocr_line.split_word(
            split_word_index=split_word_idx,
            bbox_split_offset=normalized_split_x_offset,
            character_split_index=self.split_task_word_split_idx,
        )
        ui_logger.debug(
            f"Word has now been split.\nNew Line:\n{self._current_ocr_line.text}\n"
        )
        # Expand the word bounding boxes if they were shrunk too much or didn't include all the connected pixels
        ui_logger.debug("Expanding word bounding boxes to content.")
        for word in self._current_ocr_line.items:
            word.bounding_box = word.bounding_box.expand_to_content(
                image=self._current_ocr_page.cv2_numpy_page_image,
            )

        # Shrink the word bounding boxes to fit the text
        self._current_ocr_line.refine_bounding_boxes(
            image=self._current_ocr_page.cv2_numpy_page_image, padding_px=1
        )
        ui_logger.debug("Word bounding boxes have been refined.")

        # Rematch the words in the line to the associated line ground truth in the page
        ocr_line_tuple = tuple([w.text for w in self._current_ocr_line.items])
        ground_truth_tuple = tuple(
            self._current_ocr_line.base_ground_truth_text.split(" ")
        )

        update_line_with_ground_truth(
            self._current_ocr_line,
            ocr_line_tuple=ocr_line_tuple,
            ground_truth_tuple=ground_truth_tuple,
        )

        ui_logger.debug(
            f"Line has been rematched with ground truth.\nNew Line:\n{self._current_ocr_line.text}\nGround Truth:\n{self._current_ocr_line.ground_truth_text}\n"
        )

        # if self.line_change_callback:
        #     self.line_change_callback()

        if self.page_image_change_callback:
            self.page_image_change_callback()

        self.task_type = EditorTaskType.NONE
        self.task_match_idx = None
        self.split_task_x_coordinate = -1
        self.split_task_image_width = -1

        self.redraw_ui()
        ui_logger.debug("Split task executed.")
        return

    def create_ui_word_match_widgets(self):
        ui_logger.debug("Creating UI word match widgets...")
        match_VBox = VBox()
        match_VBox.layout = Layout(padding="0px", margin="0px", flex="0 0 auto")

        image_HBox = HBox()
        ocr_HBox = HBox()
        gt_HBox = HBox()
        action_buttons_HBox = HBox()

        match_VBox.children = [
            image_HBox,
            ocr_HBox,
            gt_HBox,
            action_buttons_HBox,
        ]

        self.WordMatchingTableVBoxes.append(match_VBox)

    def load_word_match_image(self, match):
        logger.debug(f"Loading word match image for match: {match['idx']}...")
        image_HBox = self.WordMatchingTableVBoxes[match["idx"]].children[0]
        if match["img_tag_text"]:
            image_HBox.children = [
                HTML(match["img_tag_text"]),
            ]
        else:
            image_HBox.children = [
                HTML("No Image"),
            ]
        image_HBox.layout = self.basic_box_layout

    def load_word_match_text(self, match):
        logger.debug(f"Loading word match text for match: {match['idx']}...")
        # Set the word match text
        ocr_HBox = self.WordMatchingTableVBoxes[match["idx"]].children[1]
        gt_HBox = self.WordMatchingTableVBoxes[match["idx"]].children[2]

        if match["ocr_text"]:
            ocr_HBox.children = [
                get_formatted_text_html_span(
                    linecolor_css=match["ocr_text_color"],
                    text=match["ocr_text"],
                    font_family_css=self.monospace_font_name,
                )
            ]
        else:
            ocr_HBox.children = [
                get_formatted_text_html_span(
                    linecolor_css=match["ocr_text_color"],
                    text="No OCR",
                    font_family_css=self.monospace_font_name,
                )
            ]
        ocr_HBox.layout = self.basic_box_layout

        gt_width = max(len(match["gt_text"]) * 9 + 16, 100)

        gt_TextBox = Text(
            value=match["gt_text"],
            description="",
            disabled=False,
            continuous_update=False,
            style={"font-family": self.monospace_font_name},
            layout=Layout(
                width=f"{gt_width}px",
                padding="0px",
                margin="0px",
            ),
        )
        gt_HBox.children = [
            gt_TextBox,
        ]
        # listen to changes in the text box
        gt_TextBox.observe(
            lambda change, match=match: self.update_gt_text(change, match),
            names="value",
        )

    def load_word_action_buttons(self, match):
        logger.debug(f"Loading word action buttons for match: {match['idx']}...")
        action_buttons_HBox = self.WordMatchingTableVBoxes[match["idx"]].children[3]
        action_buttons_HBox_children = []

        if self.task_type != EditorTaskType.NONE:
            ui_logger.debug(
                f"Task is active: {self.task_type.name}. No Buttons will be displayed."
            )
            # If an editor task is active, don't display action buttons
            return

        delete_button = Button(
            description="X",
            layout=Layout(width="16px", padding="0px", margin="0px"),
        )
        delete_button.on_click(lambda _: self.delete_match(match))
        action_buttons_HBox_children.append(delete_button)

        word = match["word"]
        if word:
            if match["word_idx"] > 0:
                ml_button = Button(
                    description="ML",
                    layout=Layout(width="22px", padding="0px", margin="0px"),
                )
                ml_button.on_click(lambda _: self.merge_left(match))
                action_buttons_HBox_children.append(ml_button)

            if match["word_idx"] < len(self._current_ocr_line.items) - 1:
                mr_button = Button(
                    description="MR",
                    layout=Layout(width="22px", padding="0px", margin="0px"),
                )
                mr_button.on_click(lambda _: self.merge_right(match))
                action_buttons_HBox_children.append(mr_button)

            edit_bbox_button = Button(
                description="EB",
                layout=Layout(width="22px", padding="0px", margin="0px"),
            )
            edit_bbox_button.on_click(lambda _: self.start_edit_bbox_task(match))
            action_buttons_HBox_children.append(edit_bbox_button)

            split_button = Button(
                description="SP",
                layout=Layout(width="22px", padding="0px", margin="0px"),
            )
            split_button.on_click(lambda _: self.start_split_task(match))
            action_buttons_HBox_children.append(split_button)

            # Split (One Active Split/Edit Per Line):
            # <Image with Vertical Red Line>
            # <Adjust Split Left 1%> <Adjust Split Left 10%> <Adjust Split Right 1%> <Adjust Split Right 10%> <Split OK>

            # Edit Bounding Box (One Active Split/Edit Per Line):
            # <Padded Image with Red Bounding Box>
            # <Adjust Left 1%> <Adjust Left 10%>
            # <Adjust Top 1%> <Adjust Top 10%>
            # <Adjust Right 1%> <Adjust Right 10%>
            # <Adjust Bottom 1%> <Adjust Bottom 10%>
            # <Shrink to Fit>
            # <Edit OK>

        action_buttons_HBox.children = action_buttons_HBox_children
        action_buttons_HBox.layout = Layout(flex_wrap="wrap")

    def load_word_match_widgets(self, match):
        self.load_word_match_image(match)
        self.load_word_match_text(match)
        self.load_word_action_buttons(match)

    def draw_ui_word_matching_table(self):
        ui_logger.debug("Drawing UI for word matching table.")
        self.WordMatchingTableHBox = HBox()
        self.WordMatchingTableVBoxes = []

        for match in self.line_matches:
            self.create_ui_word_match_widgets()
            self.load_word_match_widgets(match)

        self.WordMatchingTableHBox.children = [
            match_vbox for match_vbox in self.WordMatchingTableVBoxes
        ]
        self.WordMatchingTableHBox.layout

    ####################################################################
    # ACTIONS
    ####################################################################

    def mark_validated(self, event=None):
        logger.debug(f"Marking line as validated: {self._current_ocr_line.text[:20]}")
        # Mark the line as validated
        if self._current_ocr_line.unmatched_ground_truth_words:
            logger.debug(
                f"Error marking line as validated: {self._current_ocr_line.text} has unmatched ground truth words."
            )
            # Display Error Message
            error_message = HTML(
                "<span style='color:red;'>Error Marking as Validated: Line has unmatched ground truth words.</span>"
            )
            self.TaskHBox.children = [error_message]
            self.TaskHBox.layout = self.basic_box_layout
            return
        if self._current_ocr_line.ground_truth_text == "":
            logger.debug(
                f"Error marking line as validated: {self._current_ocr_line.text} has no ground truth text."
            )
            # Display Error Message
            error_message = HTML(
                "<span style='color:red;'>Error Marking as Validated: Line has no ground truth text.</span>"
            )
            self.TaskHBox.children = [error_message]
            self.TaskHBox.layout = self.basic_box_layout
            return

        logger.debug("Setting line_editor_validated to True")
        self._current_ocr_line.additional_block_attributes["line_editor_validated"] = (
            True
        )

        logger.debug("Calling line_change_callback")
        if self.line_change_callback:
            self.line_change_callback()

        logger.debug("Redrawing UI")
        self.redraw_ui()

    def delete_match(self, match):
        # Delete the match from the line
        word: Word = match["word"]
        if word:
            self._current_ocr_line.remove_item(word)
            self._current_ocr_page.remove_empty_items()
        else:
            self._current_ocr_line.unmatched_ground_truth_words = [
                unmatched_gt_word
                for unmatched_gt_word in self._current_ocr_line.unmatched_ground_truth_words
                if unmatched_gt_word[0] != match["word_idx"]
                and unmatched_gt_word[1] != match["gt_text"]
            ]
        self.redraw_ui()
        if self.page_image_change_callback:
            self.page_image_change_callback()

    def update_gt_text(self, change, match):
        # Update the ground truth text in the word
        word: Word = match["word"]
        if word:
            word.ground_truth_text = change["new"]
            word.ground_truth_match_keys["match_score"] = word.fuzz_score_against(
                word.ground_truth_text
            )
            self.redraw_ui()
            if self.page_image_change_callback:
                self.page_image_change_callback()

    def copy_ocr_to_gt(self, event=None):
        # Copy all of the the OCR text into the GT text for each OCR word
        word: Word
        for word in self._current_ocr_line.items:
            word.ground_truth_text = word.text
            word.ground_truth_match_keys["match_score"] = word.fuzz_score_against(
                word.ground_truth_text
            )
        # Redraw the UI after update
        self.redraw_ui()

    def delete_line(self, event=None):
        # Delete the line from the OCR page
        self._current_ocr_page.remove_line_if_exists(self._current_ocr_line)
        self._current_ocr_page.remove_empty_items()
        self.GridVBox.children = []
        self.GridVBox.layout = Layout(display="none")
        #        if self.line_change_callback:
        #            self.line_change_callback()

        if self.page_image_change_callback:
            self.page_image_change_callback()

    def merge_left(self, match):
        word: Word = match["word"]
        word_idx = match["word_idx"]
        # Merge the word with the previous word
        if word_idx > 0:
            prev_word: Word = self._current_ocr_line.items[word_idx - 1]
            prev_word.merge(word)
            self._current_ocr_line.remove_item(word)
            self.redraw_ui()
            if self.page_image_change_callback:
                self.page_image_change_callback()

    def merge_right(self, match):
        word: Word = match["word"]
        word_idx = match["word_idx"]
        # Merge the word with the next word
        if word_idx < len(self._current_ocr_line.items) - 1:
            next_word: Word = self._current_ocr_line.items[word_idx + 1]
            word.merge(next_word)
            self._current_ocr_line.remove_item(next_word)
            self.redraw_ui()
            if self.line_change_callback:
                self.line_change_callback()
            if self.page_image_change_callback:
                self.page_image_change_callback()

    def calculate_line_matches(self):
        matches = []

        logger.debug(
            f"Calculating match components for line: {self._current_ocr_line.text[0:20]}..."
        )

        word: Word
        for word_idx, word in enumerate(self._current_ocr_line.items):
            gt_text = word.ground_truth_text or ""
            ocr_text = word.text or ""
            logger.debug(f"Word: {ocr_text} | {gt_text}")

            img_ndarray, _, _, data_src_string = get_cropped_word_image(
                img=self._current_ocr_page.cv2_numpy_page_image,
                word=word,
            )

            # Encode the cropped image as PNG and get a <img> tag string
            img_tag_text = get_html_string_from_image_src(
                data_src_string=data_src_string, height="height: 14px;"
            )

            ocr_text_color = "lightgray"
            gt_text_color = "lightgray"
            if "match_score" not in word.ground_truth_match_keys or (
                word.ground_truth_match_keys["match_score"] == 0 and gt_text == ""
            ):
                ocr_text_color = "red"
            elif word.ground_truth_match_keys["match_score"] != 100:
                ocr_text_color = "blue"
                gt_text_color = "blue"

            match = {
                "word_idx": word_idx,
                "word": word,
                "img_ndarray": img_ndarray,
                "data_src_string": data_src_string,
                "img_tag_text": img_tag_text,
                "ocr_text": ocr_text,
                "gt_text": gt_text,
                "ocr_text_color": ocr_text_color,
                "gt_text_color": gt_text_color,
            }

            matches.append(match)

        logger.debug("Adding unmatched ground truth words to matches...")

        # Insert "unmatched" ground truth words
        if self._current_ocr_line.unmatched_ground_truth_words:
            # Insert unmatched ground truth words in reverse order
            # to avoid messing up the indices of the already inserted words
            # (since we are inserting into the same list)
            for unmatched_words in reversed(
                self._current_ocr_line.unmatched_ground_truth_words
            ):
                unmatched_gt_word_idx = unmatched_words[0]
                unmatched_gt_word = unmatched_words[1]

                match = {
                    "word_idx": unmatched_gt_word_idx + 1,
                    "word": None,
                    "img_ndarray": None,
                    "data_src_string": None,
                    "img_tag_text": None,
                    "ocr_text": "",
                    "gt_text": unmatched_gt_word,
                    "ocr_text_color": "red",
                    "gt_text_color": "red",
                }

                matches.insert(unmatched_gt_word_idx + 1, match)

        logger.debug("Adding Indexer")

        # add an indexer to matches
        for idx, match in enumerate(matches):
            match["idx"] = idx

        self.line_matches = matches

        logger.debug(f"Line matching complete: {self._current_ocr_line.text[0:20]}...")
