import pathlib

# from logging import DEBUG as logging_DEBUG
from logging import getLogger

import cv2
import torch
from IPython.display import display
from ipywidgets import (
    HTML,
    BoundedIntText,
    Button,
    HBox,
    Image,
    Layout,
    Tab,
    VBox,
)

from doctr.io import DocumentFile
from doctr.models import (
    crnn_vgg16_bn,
    db_resnet50,
    detection_predictor,
    ocr_predictor,
    recognition_predictor,
)
from pd_book_tools.image_processing.cv2_processing.encoding import (
    encode_bgr_image_as_png,
)
from pd_book_tools.ocr.document import Document
from pd_book_tools.ocr.page import Page
from pd_book_tools.pgdp.pgdp_results import PGDPExport, PGDPPage

from .ipynb_page_editor import IpynbPageEditor

# Configure logging
logger = getLogger(__name__)
ui_logger = getLogger(__name__ + ".UI")


layout_no_padding_margin = Layout(padding="0px", margin="0px", flex="1 1 auto")


class IpynbLabeler:
    _current_page_idx = 0
    _total_pages = 0

    current_page_name = ""
    go_to_page_idx = 0

    overall_vbox: VBox

    # Header
    prev_button: Button
    next_button: Button
    current_page_idx_display: HTML
    current_page_name_display: HTML
    go_to_page_button: Button
    go_to_page_textbox: BoundedIntText
    header_box: HBox

    # Main Section Layout: Image to left, 'Editor' to Right
    main_hbox = HBox

    image_vbox = VBox
    editor_vbox = VBox

    # Left - Image Tabs
    image_tab: Tab
    plain_image_vbox: VBox
    ocr_image_pgh_bounding_box_vbox: VBox
    ocr_image_lines_bounding_box_vbox: VBox
    ocr_image_words_bounding_box_vbox: VBox
    ocr_image_mismatches_vbox: VBox

    plain_image: Image
    ocr_image_pgh_bounding_box: Image
    ocr_image_lines_bounding_box: Image
    ocr_image_words_bounding_box: Image
    ocr_image_mismatches: Image

    # Right - Editor Tabs
    editor_tab: Tab
    editor_ocr_text_vbox: VBox
    editor_p3_text_vbox: VBox

    matched_ocr_pages = {}

    monospace_font_name: str
    monospace_font_path: str

    pgdp_export: PGDPExport
    training_set_output_path: pathlib.Path
    validation_set_output_path: pathlib.Path

    page_indexby_name: dict
    page_indexby_nbr: dict

    ocr_models: dict
    main_ocr_predictor: ocr_predictor

    page_editor: IpynbPageEditor

    def init_font(
        self,
        monospace_font_name: str,
        monospace_font_path: pathlib.Path | str,
    ):
        self.monospace_font_name = monospace_font_name
        if isinstance(monospace_font_path, str):
            monospace_font_path = pathlib.Path(monospace_font_path)
        self.monospace_font_path = monospace_font_path
        # Inject custom CSS for the Jupyter environment
        css = f"""
        @font-face {{
            font-family: '{self.monospace_font_name}';
            src: url('{self.monospace_font_path}') format('truetype');
        }}
        """

        # input, textarea {{
        #     font-family: '{self.monospace_font_path}', monospace !important;
        #     font-size: 12px !important;
        # }}

        display(HTML(f"<style>{css}</style>"))

    def init_header_ui(self):
        self.prev_button = Button(description="Previous")
        self.next_button = Button(description="Next")
        self.current_page_idx_display = HTML("")
        self.current_page_name_display = HTML("")
        self.go_to_page_button = Button(description="Go to Page #")
        self.go_to_page_textbox = BoundedIntText()
        self.go_to_page_textbox.layout = Layout(width="65px")

        self.save_page_changes_button = Button(
            description="Save OCR Changes",
        )
        # TODO: Add Onclick to Save OCR as dict to a file

        self.export_training_button = Button(
            description="Export to ML Training",
        )
        self.export_training_button.on_click(self.save_training_button)

        self.export_validation_button = Button(
            description="Export to ML Validation",
        )
        self.export_validation_button.on_click(self.save_validations_button)

        self.reset_ocr_button = Button(description="Reset Page OCR")

        def reset_ocr(event=None):
            # Reset the OCR for the current page
            self.run_ocr(force_refresh_ocr=True)
            self.refresh_ui()

        self.reset_ocr_button.on_click(reset_ocr)

        self.header_box = VBox(
            [
                HBox(
                    [
                        self.prev_button,
                        self.current_page_idx_display,
                        self.current_page_name_display,
                        self.next_button,
                        self.go_to_page_button,
                        self.go_to_page_textbox,
                    ]
                ),
                HBox(
                    [
                        self.save_page_changes_button,
                        self.export_training_button,
                        self.export_validation_button,
                        self.reset_ocr_button,
                    ]
                ),
            ]
        )
        self.header_box.layout = Layout(
            flex="0 0 auto",
            padding="0px",
            margin="0px",
            width="100%",
        )

        self.prev_button.on_click(self.prev_page)
        self.next_button.on_click(self.next_page)
        self.go_to_page_button.on_click(self.go_to_page)

    def init_main_ui(self):
        self.plain_image = Image(
            layout=Layout(min_width="300px", max_height="900px", align_self="baseline")
        )
        self.plain_image_vbox = VBox([self.plain_image])

        self.ocr_image_pgh_bounding_box = Image(
            layout=Layout(min_width="300px", max_height="900px", align_self="baseline")
        )
        self.ocr_image_pgh_bounding_box_vbox = VBox(
            [self.ocr_image_pgh_bounding_box], layout={"overflow": "visible"}
        )

        self.ocr_image_lines_bounding_box = Image(
            layout=Layout(min_width="300px", max_height="900px", align_self="baseline")
        )
        self.ocr_image_lines_bounding_box_vbox = VBox(
            [self.ocr_image_lines_bounding_box], layout={"overflow": "visible"}
        )

        self.ocr_image_words_bounding_box = Image(
            layout=Layout(min_width="300px", max_height="900px", align_self="baseline")
        )
        self.ocr_image_words_bounding_box_vbox = VBox(
            [self.ocr_image_words_bounding_box], layout={"overflow": "visible"}
        )

        self.ocr_image_mismatches = Image(
            layout=Layout(min_width="300px", max_height="900px", align_self="baseline")
        )
        self.ocr_image_mismatches_vbox = VBox(
            [self.ocr_image_mismatches], layout={"overflow": "visible"}
        )

        image_tabs = [
            (
                "Mismatches",
                self.ocr_image_mismatches_vbox,
            ),
            (
                "Original",
                self.plain_image_vbox,
            ),
            (
                "Paragraphs",
                self.ocr_image_pgh_bounding_box_vbox,
            ),
            (
                "Lines",
                self.ocr_image_lines_bounding_box_vbox,
            ),
            (
                "Words",
                self.ocr_image_words_bounding_box_vbox,
            ),
        ]
        image_tab_titles, image_tab_boxes = zip(*image_tabs)

        self.image_tab = Tab(children=image_tab_boxes)
        self.image_tab.titles = image_tab_titles

        self.image_vbox = VBox(
            [
                self.image_tab,
            ]
        )
        self.image_vbox.layout = Layout(flex="0 0 30%")

        try:
            current_pgdp_page = self.current_pgdp_page
        except KeyError:
            current_pgdp_page = None
        try:
            current_ocr_page = self.current_ocr_page
        except KeyError:
            current_ocr_page = None

        def page_image_change_callback():
            self.current_ocr_page.refresh_page_images()
            self.update_images()

        self.page_editor = IpynbPageEditor(
            current_pgdp_page,
            current_ocr_page,
            self.monospace_font_name,
            self.monospace_font_path,
            page_image_change_callback=page_image_change_callback,
        )

        self.editor_line_matching_vbox = self.page_editor.editor_line_matching_vbox
        self.editor_ocr_text_vbox = VBox()
        self.editor_p3_text_vbox = VBox()

        editor_tabs = [
            (
                "Matching",
                self.editor_line_matching_vbox,
            ),
            (
                "OCR Text",
                self.editor_ocr_text_vbox,
            ),
            (
                "PGDP P3 Text",
                self.editor_p3_text_vbox,
            ),
        ]
        editor_tab_titles, editor_tab_boxes = zip(*editor_tabs)

        self.editor_tab = Tab(children=editor_tab_boxes)
        self.editor_tab.titles = editor_tab_titles

        self.editor_vbox = VBox(
            [
                self.editor_tab,
            ]
        )
        self.editor_vbox.layout = Layout(flex="1 1 auto", overflow="scroll")

        self.main_hbox = HBox(
            [
                self.image_vbox,
                self.editor_vbox,
            ]
        )
        self.main_hbox.layout = Layout(height="100%", flex="0 0 auto")

    def init_footer_ui(self):
        self.footer_hbox = HBox()

    def init_ocr(self):
        if self.doctr_predictor:
            # Use the provided doctr predictor if available
            self.main_ocr_predictor = self.doctr_predictor
            return
        # Otherwise, use the default doctr models

        # Check if GPU is available
        device, device_nbr = (
            ("cuda", "cuda:0") if torch.cuda.is_available() else ("cpu", "cpu")
        )
        logger.info(f"Using {device} for OCR")

        det_model = db_resnet50(pretrained=True).to(device)
        reco_model = crnn_vgg16_bn(pretrained=True, pretrained_backbone=True).to(device)

        full_predictor = ocr_predictor(
            det_arch=det_model,
            reco_arch=reco_model,
            pretrained=True,
            assume_straight_pages=True,
            disable_crop_orientation=True,
        )

        det_predictor = detection_predictor(
            arch=det_model,
            pretrained=True,
            assume_straight_pages=True,
        )

        reco_predictor = recognition_predictor(
            arch=reco_model,
            pretrained=True,
        )

        full_predictor.det_predictor = det_predictor
        full_predictor.reco_predictor = reco_predictor

        self.main_ocr_predictor = full_predictor

    def __init__(
        self,
        pgdp_export: PGDPExport,
        training_set_output_path: pathlib.Path | str,
        validation_set_output_path: pathlib.Path | str,
        monospace_font_name: str,
        monospace_font_path: pathlib.Path | str,
        start_page_name="",
        start_page_idx=0,
        doctr_predictor=None,
    ):
        self.doctr_predictor = doctr_predictor
        self.pgdp_export = pgdp_export

        if isinstance(training_set_output_path, str):
            training_set_output_path = pathlib.Path(training_set_output_path)
        self.training_set_output_path = training_set_output_path
        if not self.training_set_output_path.exists():
            self.training_set_output_path.mkdir(parents=True, exist_ok=True)

        if isinstance(validation_set_output_path, str):
            validation_set_output_path = pathlib.Path(validation_set_output_path)
        self.validation_set_output_path = validation_set_output_path
        if not self.validation_set_output_path.exists():
            self.validation_set_output_path.mkdir(parents=True, exist_ok=True)

        self.page_indexby_name = {
            item.png_file: i for i, item in enumerate(self.pgdp_export.pages)
        }
        self.page_indexby_nbr = {
            i: item.png_file for i, item in enumerate(self.pgdp_export.pages)
        }

        self._total_pages = len(self.pgdp_export.pages) - 1

        if start_page_name:
            new_idx = self.page_indexby_name.get(start_page_name, -1)
            if new_idx > 0 and new_idx < len(self.total_pages):
                self._current_page_idx = start_page_idx
                self.current_page_name = pathlib.Path(
                    self.current_pgdp_page.png_file
                ).stem
        elif start_page_idx > 0 and start_page_idx < self.total_pages:
            self._current_page_idx = start_page_idx
            self.current_page_name = pathlib.Path(self.current_pgdp_page.png_file).stem

        self.init_ocr()
        self.init_font(monospace_font_name, monospace_font_path)
        self.init_header_ui()
        self.init_main_ui()
        self.init_footer_ui()

        self.overall_vbox = VBox(
            [
                HTML(f"<style>{self._jupyter_css()}</style>"),
                self.header_box,
                self.main_hbox,
                self.footer_hbox,
            ]
        )

        self.refresh_ui()
        self.display()

    @property
    def current_page_idx(self):
        return self._current_page_idx

    @current_page_idx.setter
    def current_page_idx(self, value):
        self._current_page_idx = value
        self.current_page_name = pathlib.Path(self.current_pgdp_page.png_file).stem
        self.refresh_ui()

    @property
    def total_pages(self):
        return self._total_pages

    @total_pages.setter
    def total_pages(self, value):
        self._total_pages = value
        self.go_to_page_textbox.max = value
        self.refresh_ui()

    @property
    def current_pgdp_page(self) -> PGDPPage:
        return self.pgdp_export.pages[self.current_page_idx]

    @property
    def current_ocr_page(self) -> Page:
        return self.matched_ocr_pages[self.current_page_idx]["page"]

    def _jupyter_css(self):
        # Inject custom CSS for the Jupyter environment
        css = f"""
        @font-face {{
            font-family: '{self.monospace_font_name}';
            src: url('{self.monospace_font_path}') format('truetype');
        }}

        input, textarea {{
            font-family: '{self.monospace_font_name}', monospace !important;
            font-size: 12px !important;
        }}
        """
        logger.debug("Custom CSS:\n" + css)
        return css

    def update_header_elements(self):
        self.current_page_idx_display.value = f" #{self.current_page_idx} "
        self.current_page_name_display.value = f" Page: {self.current_page_name} "
        self.go_to_page_textbox.min = 0
        self.go_to_page_textbox.max = self.total_pages
        self.go_to_page_textbox.value = self.current_page_idx

    def update_images(self):
        self.reload_page_images_ui()

        # w = self.matched_ocr_pages[self.current_page_idx]["width"]
        # h = self.matched_ocr_pages[self.current_page_idx]["height"]
        # h = int((w / self.matched_ocr_pages[self.current_page_idx]["width"]) * h)
        # w = min(400, w)

        self.plain_image.value = self.matched_ocr_pages[self.current_page_idx][
            "page_image"
        ]
        # self.plain_image.width = w
        # self.plain_image.height = h

        self.ocr_image_pgh_bounding_box.value = self.matched_ocr_pages[
            self.current_page_idx
        ]["ocr_image_pgh_bounding_box"]
        # self.ocr_image_pgh_bounding_box.width = w
        # self.ocr_image_pgh_bounding_box.height = h

        self.ocr_image_lines_bounding_box.value = self.matched_ocr_pages[
            self.current_page_idx
        ]["ocr_image_lines_bounding_box"]
        # self.ocr_image_lines_bounding_box.width = w
        # self.ocr_image_lines_bounding_box.height = h

        self.ocr_image_words_bounding_box.value = self.matched_ocr_pages[
            self.current_page_idx
        ]["ocr_image_words_bounding_box"]
        # self.ocr_image_words_bounding_box.width = w
        # self.ocr_image_words_bounding_box.height = h

        self.ocr_image_mismatches.value = self.matched_ocr_pages[self.current_page_idx][
            "ocr_image_mismatches"
        ]
        # self.ocr_image_mismatches.width = w
        # self.ocr_image_mismatches.height = h

    def update_pgdp_text(self):
        html_lines = [
            f"""
            <tr>
                <td><span style='font-family:{self.monospace_font_name}; font-size: 12px;'>{line_idx}</span></td>
                <td><span style='font-family:{self.monospace_font_name}; font-size: 12px;'>{line}</span></td>
            </tr>
            """
            for line_idx, line in self.current_pgdp_page.processed_lines
        ]
        html_lines.insert(
            0,
            "<table>",
        )
        html_lines.append("</div></table>")
        self.editor_p3_text_vbox.children = [
            HTML("\n".join(html_lines)),
        ]

    def update_ocr_text(self):
        logger.debug("Current OCR Text:\n" + self.current_ocr_page.text)
        html_lines = [
            f"""
            <tr>
                <td><span style='font-family:{self.monospace_font_name}; font-size: 12px;'>{line_idx}</span></td>
                <td><span style='font-family:{self.monospace_font_name}; font-size: 12px;'>{line}</span></td>
            </tr>
            """
            for line_idx, line in enumerate(
                self.current_ocr_page.text.splitlines(keepends=True)
            )
        ]
        html_lines.insert(
            0,
            "<table>",
        )
        html_lines.append("</table>")
        self.editor_ocr_text_vbox.children = [
            HTML("\n".join(html_lines)),
        ]

    def update_text(self):
        self.update_ocr_text()
        self.update_pgdp_text()
        self.page_editor.update_line_matches(
            self.current_pgdp_page, self.current_ocr_page
        )
        # self.update_line_matches()

    def refresh_ui(self):
        self.update_header_elements()
        self.run_ocr()
        self.update_images()
        self.update_text()

    def save_validations_button(self, event=None):
        # Save the current page
        prefix = self.pgdp_export.project_id + "_" + str(self.current_page_idx)
        self.current_ocr_page.convert_to_training_set(
            output_path=self.validation_set_output_path,
            prefix=prefix,
        )

    def save_training_button(self, event=None):
        # Save the current page
        prefix = self.pgdp_export.project_id + "_" + str(self.current_page_idx)
        self.current_ocr_page.convert_to_training_set(
            output_path=self.training_set_output_path,
            prefix=prefix,
        )

    # Navigation Buttons
    def prev_page(self, event=None):
        if self.current_page_idx >= 1:
            self.current_page_idx -= 1
            self.refresh_ui()

    def next_page(self, event=None):
        if self.current_page_idx < self.total_pages:
            self.current_page_idx += 1
            self.refresh_ui()

    def go_to_page(self, event=None):
        go_to_page_idx = self.go_to_page_textbox.value
        if go_to_page_idx < self.total_pages and go_to_page_idx >= 0:
            self.current_page_idx = go_to_page_idx
            self.refresh_ui()

    def reload_page_images_ui(self):
        ocr_page: Page = self.matched_ocr_pages[self.current_page_idx]["page"]
        self.matched_ocr_pages[self.current_page_idx] = {
            **self.matched_ocr_pages[self.current_page_idx],
            "width": ocr_page.cv2_numpy_page_image.shape[1],
            "height": ocr_page.cv2_numpy_page_image.shape[0],
            "page_image": encode_bgr_image_as_png(ocr_page.cv2_numpy_page_image),
            "ocr_image_words_bounding_box": encode_bgr_image_as_png(
                ocr_page.cv2_numpy_page_image_word_with_bboxes
            ),
            "ocr_image_mismatches": encode_bgr_image_as_png(
                ocr_page.cv2_numpy_page_image_matched_word_with_colors
            ),
            "ocr_image_lines_bounding_box": encode_bgr_image_as_png(
                ocr_page.cv2_numpy_page_image_line_with_bboxes
            ),
            "ocr_image_pgh_bounding_box": encode_bgr_image_as_png(
                ocr_page.cv2_numpy_page_image_paragraph_with_bboxes
            ),
        }

    def run_ocr(self, force_refresh_ocr=False):
        """Run OCR or get cached or new matched OCR page and update
        docTR_ocr_cv2_image runs multiple models on the image and returns a list of results.
        """
        if (
            self.current_page_idx not in self.matched_ocr_pages.keys()
            or force_refresh_ocr
        ):
            source_image = self.current_pgdp_page.png_full_path

            doctr_image = DocumentFile.from_images(source_image)
            doctr_result = self.main_ocr_predictor(doctr_image)
            docTR_output = doctr_result.export()

            ocr_doc = Document.from_doctr_output(docTR_output, source_image)
            # Always 1 page per OCR in this case
            ocr_page: Page = ocr_doc.pages[0]
            ocr_page.cv2_numpy_page_image = cv2.imread(source_image)

            # expand bboxes of each word to capture missing pixels
            # Expand the word bounding boxes if they were shrunk too much or didn't include all the connected pixels
            ui_logger.debug("Expanding word bounding boxes to content.")
            for word in ocr_page.words:
                word.bounding_box = word.bounding_box.expand_to_content(
                    image=ocr_page.cv2_numpy_page_image
                )

            ocr_page.refine_bounding_boxes(padding_px=1)
            ocr_page.reorganize_page()
            ocr_page.add_ground_truth(self.current_pgdp_page.processed_page_text)

            self.matched_ocr_pages[self.current_page_idx] = {
                "page": ocr_page,
            }

    def display(self):
        # Inject custom CSS for the Jupyter environment monospacing the fonts
        display(self.overall_vbox)
