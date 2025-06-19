import pathlib

# from logging import DEBUG as logging_DEBUG
from logging import getLogger
from textwrap import dedent
from typing import Optional

import cv2
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

from shutil import copyfile

from doctr.models.predictor.pytorch import OCRPredictor

from pd_book_tools.image_processing.cv2_processing.encoding import (
    encode_bgr_image_as_png,
)
from pd_book_tools.ocr.document import Document
from pd_book_tools.ocr.page import Page
from pd_book_tools.pgdp.pgdp_results import PGDPExport, PGDPPage
from pd_book_tools.ocr.cv2_doctr import doctr_ocr_cv2_image, get_default_doctr_predictor


from .ipynb_page_editor import IpynbPageEditor

# Configure logging
logger = getLogger(__name__)
ui_logger = getLogger(__name__ + ".UI")


layout_no_padding_margin = Layout(padding="0px", margin="0px", flex="1 1 auto")


class IpynbLabeler:
    _current_page_idx: int = 0
    _total_pages: int = 0

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
    monospace_font_path: pathlib.Path

    doctr_predictor: Optional[OCRPredictor] = None
    pgdp_export: PGDPExport
    labeled_ocr_path: pathlib.Path
    training_set_output_path: pathlib.Path
    validation_set_output_path: pathlib.Path

    page_indexby_name: dict
    page_indexby_nbr: dict

    ocr_models: dict
    main_ocr_predictor: OCRPredictor

    page_editor: IpynbPageEditor

    def init_font(
        self,
        monospace_font_name: str,
        monospace_font_path: pathlib.Path | str | None = None,
    ):
        self.monospace_font_name = monospace_font_name
        if isinstance(monospace_font_path, str):
            monospace_font_path = pathlib.Path(monospace_font_path)
        if monospace_font_path is not None and pathlib.Path.exists(monospace_font_path):
            self.monospace_font_path = monospace_font_path
            # Inject custom CSS for font definition in the Jupyter environment
            css = dedent(
                f"""
                @font-face {{
                    font-family: '{self.monospace_font_name}';
                    src: url('{self.monospace_font_path}') format('truetype');
                }}
                """
            )
        elif monospace_font_name:
            # If no path is provided, use the font name to define the font
            css = dedent(
                f"""
                @font-face {{
                    font-family: '{self.monospace_font_name}';
                }}
                """
            )
        else:
            # Fallback to a default monospace font if no path is provided
            css = dedent(
                """
                @font-face {{
                    font-family: 'monospace';
                }}
                """
            )
        display(HTML(f"<style>{css}</style>"))

    def init_header_ui(self):
        self.prev_button = Button(description="Previous")
        self.next_button = Button(description="Next")
        self.current_page_idx_display = HTML("")
        self.current_page_name_display = HTML("")
        self.go_to_page_button = Button(description="Go to Page #")
        self.go_to_page_textbox = BoundedIntText()
        self.go_to_page_textbox.layout = Layout(width="65px")

        self.save_ocr_to_file_button = Button(
            description="Save OCR to File",
        )
        # expand button text to fill only what it needs
        self.save_ocr_to_file_button.style.button_width = "auto"
        self.save_ocr_to_file_button.layout = Layout(
            width="auto",
        )
        self.save_ocr_to_file_button.on_click(
            lambda event: self.export_ocr_document(event)
        )

        self.reload_ocr_from_file_button = Button(
            description="Reload OCR from File",
        )
        self.reload_ocr_from_file_button.layout = Layout(
            width="auto",
        )

        def reload_ocr_from_file(event):
            self.import_ocr_document()
            self.refresh_ui()

        self.reload_ocr_from_file_button.on_click(
            lambda event: reload_ocr_from_file(event)
        )

        self.reset_ocr_button = Button(description="Reset OCR")
        self.reset_ocr_button.layout = Layout(
            width="auto",
        )
        self.reset_ocr_button.on_click(lambda event: self.reset_ocr())

        self.export_training_button = Button(
            description="Export Training Set",
        )
        self.export_training_button.on_click(lambda event: self.export_training(event))
        self.export_training_button.layout = Layout(
            width="auto",
        )

        self.export_validation_button = Button(
            description="Export Validation Set",
        )
        self.export_validation_button.on_click(self.export_validations)
        self.export_validation_button.layout = Layout(
            width="auto",
        )

        self.expand_and_refine_all_bboxes_button = Button(
            description="Expand & Refine All BBoxes",
        )
        self.expand_and_refine_all_bboxes_button.layout = Layout(
            width="auto",
        )
        self.expand_and_refine_all_bboxes_button.on_click(
            lambda event: self.expand_and_refine_all_bboxes()
        )
        self.refine_all_bboxes_button = Button(
            description="Refine All BBoxes",
        )
        self.refine_all_bboxes_button.layout = Layout(
            width="auto",
        )
        self.refine_all_bboxes_button.on_click(lambda event: self.refine_all_bboxes())

        self.refresh_page_images_button = Button(
            description="Refresh Page Images",
        )
        self.refresh_page_images_button.layout = Layout(
            width="auto",
        )
        self.refresh_page_images_button.on_click(
            lambda event: self.refresh_page_images()
        )

        self.refresh_all_line_images_button = Button(
            description="Refresh All Line Images",
        )
        self.refresh_all_line_images_button.layout = Layout(
            width="auto",
        )
        self.refresh_all_line_images_button.on_click(
            lambda event: self.refresh_all_line_images()
        )

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
                        self.save_ocr_to_file_button,
                        self.reload_ocr_from_file_button,
                        self.reset_ocr_button,
                        self.export_training_button,
                        self.export_validation_button,
                    ]
                ),
                HBox(
                    [
                        self.expand_and_refine_all_bboxes_button,
                        self.refine_all_bboxes_button,
                        self.refresh_page_images_button,
                        self.refresh_all_line_images_button,
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

    def init_image_ui(self):
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

    def init_main_ui(self):
        self.init_image_ui()

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

    def init_ocr_doctr_predictor(self):
        if self.doctr_predictor:
            # Use the provided doctr predictor if provided
            self.main_ocr_predictor = self.doctr_predictor
            return

        # Otherwise, use the default doctr models (not fine-tuned)
        self.main_ocr_predictor = get_default_doctr_predictor() # type: ignore[assignment]

    def create_path(self, str_or_path: pathlib.Path | str) -> pathlib.Path:
        if isinstance(str_or_path, str):
            str_or_path = pathlib.Path(str_or_path)
        if not str_or_path.exists():
            str_or_path.mkdir(parents=True, exist_ok=True)
        return str_or_path

    def __init__(
        self,
        pgdp_export: PGDPExport,
        labeled_ocr_path: pathlib.Path | str,
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

        self.labeled_ocr_path: pathlib.Path = self.create_path(
            labeled_ocr_path,
        )
        self.training_set_output_path: pathlib.Path = self.create_path(
            training_set_output_path,
        )
        self.validation_set_output_path: pathlib.Path = self.create_path(
            validation_set_output_path,
        )

        self.page_indexby_name = {
            item.png_file: i for i, item in enumerate(self.pgdp_export.pages)
        }
        self.page_indexby_nbr = {
            i: item.png_file for i, item in enumerate(self.pgdp_export.pages)
        }

        self._total_pages = len(self.pgdp_export.pages) - 1

        if start_page_name:
            new_idx = self.page_indexby_name.get(start_page_name, -1)
            if new_idx > 0 and new_idx < self.total_pages:
                self._current_page_idx = start_page_idx
                self.current_page_name = pathlib.Path(
                    self.current_pgdp_page.png_file
                ).stem
        elif start_page_idx > 0 and start_page_idx < self.total_pages:
            self._current_page_idx = start_page_idx
            self.current_page_name = pathlib.Path(self.current_pgdp_page.png_file).stem

        self.init_ocr_doctr_predictor()
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
    def total_pages(self) -> int:
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

    def refresh_ui(self):
        ui_logger.debug("Refreshing UI for page index: " + str(self.current_page_idx))

        self.update_header_elements()
        self.run_ocr()
        self.update_images()
        self.update_text()

        self.page_editor.update_line_matches(
            self.current_pgdp_page, self.current_ocr_page
        )
        ui_logger.debug("UI refreshed for page index: " + str(self.current_page_idx))

    # Navigation Buttons
    def prev_page(self, event=None):
        ui_logger.debug("Going to previous page")
        if self.current_page_idx >= 1:
            self.current_page_idx = self.current_page_idx - 1

    def next_page(self, event=None):
        ui_logger.debug("Going to next page")
        if self.current_page_idx < self.total_pages:
            self.current_page_idx = self.current_page_idx + 1

    def go_to_page(self, event=None):
        go_to_page_idx = self.go_to_page_textbox.value
        ui_logger.debug(f"Going to page index: {go_to_page_idx}")
        if go_to_page_idx < self.total_pages and go_to_page_idx >= 0:
            self.current_page_idx = go_to_page_idx

    def reload_page_images_ui(self):
        logger.debug(f"Reloading page images for page index: {self.current_page_idx}")
        ocr_page: Page = self.matched_ocr_pages[self.current_page_idx]["page"]
        if ocr_page.cv2_numpy_page_image is None:
            raise ValueError(
                "Current OCR page does not have a valid image."
            )
        self.matched_ocr_pages[self.current_page_idx] = {
            **self.matched_ocr_pages[self.current_page_idx],
            "width": ocr_page.cv2_numpy_page_image.shape[1],
            "height": ocr_page.cv2_numpy_page_image.shape[0],
            "page_image": encode_bgr_image_as_png(ocr_page.cv2_numpy_page_image),
            "ocr_image_words_bounding_box": encode_bgr_image_as_png(
                ocr_page.cv2_numpy_page_image_word_with_bboxes
            ) if ocr_page.cv2_numpy_page_image_word_with_bboxes is not None else None,
            "ocr_image_mismatches": encode_bgr_image_as_png(
                ocr_page.cv2_numpy_page_image_matched_word_with_colors
            ) if ocr_page.cv2_numpy_page_image_matched_word_with_colors is not None else None,
            "ocr_image_lines_bounding_box": encode_bgr_image_as_png(
                ocr_page.cv2_numpy_page_image_line_with_bboxes
            ) if ocr_page.cv2_numpy_page_image_line_with_bboxes is not None else None,
            "ocr_image_pgh_bounding_box": encode_bgr_image_as_png(
                ocr_page.cv2_numpy_page_image_paragraph_with_bboxes
            ) if ocr_page.cv2_numpy_page_image_paragraph_with_bboxes is not None else None,
        }

    def refresh_page_images(self):
        """Refresh the page images for the current page."""
        ui_logger.debug(f"Refreshing page images for page index: {self.current_page_idx}")
        ocr_page: Page = self.matched_ocr_pages[self.current_page_idx]["page"]
        
        # Regenerate all the page images with bounding boxes
        ocr_page.refresh_page_images()
        
        # Update the UI with the new images
        self.reload_page_images_ui()
        self.update_images()

    def refresh_all_line_images(self):
        """Refresh all line images for the current page."""
        ui_logger.debug(f"Refreshing all line images for page index: {self.current_page_idx}")
        # TODO
        # iterate over the line editors and have them regenerate their images
        ocr_page: Page = self.matched_ocr_pages[self.current_page_idx]["page"]
        if ocr_page.cv2_numpy_page_image is None:
            raise ValueError(
                "Current OCR page does not have a valid image to refresh line images."
            )
        

    def expand_and_refine_all_bboxes(self):
        """Expand the bounding boxes of all words in the current OCR page."""
        ocr_page: Page = self.matched_ocr_pages[self.current_page_idx]["page"]

        ui_logger.debug("Expanding all word bounding boxes to content and beyond.")
        for word in ocr_page.words:
            logger.debug(
                f"Refining word bounding box for word: {word.text} with bbox: {word.bounding_box}"
            )
            if ocr_page.cv2_numpy_page_image is None:
                raise ValueError(
                    "Current OCR page does not have a valid image to refine bounding boxes."
                )
            word.bounding_box = word.bounding_box.crop_bottom(
                image=ocr_page.cv2_numpy_page_image
            )
            word.bounding_box = word.bounding_box.expand_to_content(
                image=ocr_page.cv2_numpy_page_image
            )

        ocr_page.refine_bounding_boxes(padding_px=2)
        self.refresh_ui()

    def refine_all_bboxes(self):
        """Refine the bounding boxes of all words in the current OCR page."""
        ocr_page: Page = self.matched_ocr_pages[self.current_page_idx]["page"]
        ocr_page.refine_bounding_boxes(padding_px=2)
        self.refresh_ui()

    def run_ocr(self, force_refresh_ocr=False):
        """Run OCR or get saved OCR document dict and update the current page
        If the OCR document is already saved, it will be imported and used.
        If force_refresh_ocr is True, it will re-run the OCR even if the document exists.
        """
        ui_logger.debug(
            f"Running OCR for page index: {self.current_page_idx}, force_refresh_ocr: {force_refresh_ocr}"
        )
        # Import the saved label data if it exists
        self.import_ocr_document()

        if (
            self.current_page_idx not in self.matched_ocr_pages.keys()
            or force_refresh_ocr
        ):
            source_image = self.current_pgdp_page.png_full_path

            cv2_numpy_image = cv2.imread(str(source_image.resolve()))

            # Always 1 page per OCR in this case
            ocr_page: Page = doctr_ocr_cv2_image(
                image=cv2_numpy_image,
                source_image=str(source_image),
                predictor=self.main_ocr_predictor,
            )
            ocr_page.cv2_numpy_page_image = cv2.imread(str(source_image.resolve()))

            ui_logger.debug(
                f"OCR Completed. Page text length: {len(ocr_page.text)} characters."
            )

            ocr_page.reorganize_page()

            ui_logger.debug("OCR Reorganized.")

            ocr_page.add_ground_truth(self.current_pgdp_page.processed_page_text)
            ui_logger.debug("Ground Truth Added.")

            self.matched_ocr_pages[self.current_page_idx] = {
                "page": ocr_page,
            }

    def reset_ocr(self, event=None):
        # Reset the OCR for the current page and refresh the UI
        self.run_ocr(force_refresh_ocr=True)
        self.refresh_ui()

    @property
    def export_prefix(self):
        return f"{self.pgdp_export.project_id}_{self.current_page_idx}"

    def export_validations(self, event=None):
        # Save the current page
        prefix = self.export_prefix
        self.current_ocr_page.convert_to_training_set(
            output_path=self.validation_set_output_path,
            prefix=prefix,
        )

    def export_training(self, event=None):
        # Save the current page
        prefix = self.export_prefix
        self.current_ocr_page.convert_to_training_set(
            output_path=self.training_set_output_path,
            prefix=prefix,
        )

    def export_ocr_document(self, event=None):
        """Export a dict of the current OCR document to a file"""
        if self.current_page_idx not in self.matched_ocr_pages:
            logger.error("No OCR page to save.")
            return

        ocr_page: Page = self.matched_ocr_pages[self.current_page_idx]["page"]
        ocr_image_path = self.current_pgdp_page.png_full_path

        # copy the image to the labeled OCR path
        target_image_path = pathlib.Path(
            self.labeled_ocr_path, f"{self.export_prefix}.png"
        )

        ocr_document: Document = Document(
            pages=[ocr_page],
            source_lib="doctr-pgdp-labeled",
            source_path=target_image_path,
        )

        target_path = pathlib.Path(self.labeled_ocr_path, f"{self.export_prefix}.json")

        logger.info(f"Exporting OCR document to {target_path}")
        ocr_document.to_json_file(target_path)

        logger.info(f"Copying OCR image to {target_image_path}")
        copyfile(ocr_image_path, target_image_path)

    def import_ocr_document(self):
        """Import an OCR document from a JSON file and update the current page"""
        ocr_json_path = pathlib.Path(
            self.labeled_ocr_path, f"{self.export_prefix}.json"
        )
        logger.debug(f"Importing OCR document from {ocr_json_path}")

        if not ocr_json_path.exists():
            logger.error(f"OCR JSON file does not exist: {ocr_json_path}")
            return

        ocr_document: Document = Document.from_json_file(ocr_json_path)
        if len(ocr_document.pages) != 1:
            logger.error("OCR document must contain exactly one page.")
            raise ValueError("OCR document must contain exactly one page.")
            return

        ocr_page: Page = ocr_document.pages[0]
        source_image = ocr_document.source_path
        if not source_image:
            logger.error("OCR document does not contain a source image path.")
            raise ValueError("OCR document does not contain a source image path.")
            return
        ocr_page.cv2_numpy_page_image = cv2.imread(str(source_image.resolve()))

        self.matched_ocr_pages[self.current_page_idx] = {
            "page": ocr_page,
        }

    def display(self):
        # Inject custom CSS for the Jupyter environment monospacing the fonts
        display(self.overall_vbox)
