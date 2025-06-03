import pathlib
from enum import Enum
from logging import DEBUG as logging_DEBUG
from logging import getLogger

from ipywidgets import Layout, RadioButtons, VBox

from pd_book_tools.ocr.page import Page
from pd_book_tools.pgdp.pgdp_results import PGDPPage

from .ipynb_line_editor import IpynbLineEditor

# Configure logging
logger = getLogger(__name__)
ui_logger = getLogger(__name__ + ".UI")


class LineMatching(Enum):
    SHOW_ALL_LINES = 1
    SHOW_ONLY_MISMATCHES = 2
    SHOW_ONLY_UNVALIDATED_MISMATCHES = 3


class IpynbPageEditor:
    """
    UI for adding/removing lines within in a page
    """

    line_matching_configuration: LineMatching.SHOW_ALL_LINES

    _current_pgdp_page: PGDPPage | None
    _current_ocr_page: Page | None

    page_image_change_callback: callable = None

    line_editors: list[IpynbLineEditor]

    def _observe_show_exact_line_matches(self, change=None):
        logger.debug(f"Radio Button Changed: {str(change)}")
        if change is None or change["type"] != "change":
            logger.debug("Not a change event.")
            return
        if change["name"] != "value":
            logger.debug("Not a 'value' change event.")
            return
        if change["new"] is None:
            logger.debug("New value is None.")
            return
        if change["new"] == self.line_matching_configuration.value:
            logger.debug("New value is the same as the old value.")
            return

        self.line_matching_configuration = change["new"]
        self.rebuild_content_ui()

    editor_line_matching_vbox_header: VBox
    show_exact_line_matches_radiobuttons: RadioButtons
    editor_line_matching_vbox_content: VBox
    editor_line_matching_vbox_footer: VBox

    editor_line_matching_vbox: VBox

    monospace_font_name: str
    monospace_font_path: pathlib.Path

    def init_font(
        self,
        monospace_font_name: str,
        monospace_font_path: pathlib.Path | str,
    ):
        self.monospace_font_name = monospace_font_name
        if isinstance(monospace_font_path, str):
            monospace_font_path = pathlib.Path(monospace_font_path)
        self.monospace_font_path = monospace_font_path

    def init_header_ui(self):
        self.editor_line_matching_vbox_header = VBox()
        self.editor_line_matching_vbox_header.layout = Layout(
            margin="0px",
            padding="0px",
            flex="1 0 auto",
        )

        # Create radio button in header for exact matches
        self.show_exact_line_matches_radiobuttons = RadioButtons(
            options={
                "All Lines": LineMatching.SHOW_ALL_LINES,
                "Only Mismatches": LineMatching.SHOW_ONLY_MISMATCHES,
                "Only Unvalidated Mismatches": LineMatching.SHOW_ONLY_UNVALIDATED_MISMATCHES,
            },
            value=self.line_matching_configuration,
            description="",
            disabled=False,
            orientation="horizontal",
            layout=Layout(width="max-content"),
        )
        self.show_exact_line_matches_radiobuttons.observe(
            handler=self._observe_show_exact_line_matches,
        )

        editor_line_matching_vbox_header_children = [
            self.show_exact_line_matches_radiobuttons,
        ]

        self.editor_line_matching_vbox_header.children = (
            editor_line_matching_vbox_header_children
        )

    def init_footer_ui(self):
        self.editor_line_matching_vbox_footer = VBox()
        self.editor_line_matching_vbox_footer.layout = Layout(
            margin="0px",
            padding="0px",
            flex="1 0 auto",
        )

    def init_ui(self):
        self.init_header_ui()
        self.init_footer_ui()

        self.editor_line_matching_vbox_content = VBox()
        self.editor_line_matching_vbox_content.layout = Layout(
            margin="0px",
            padding="0px",
            overflow="scroll",
            flex="0 0 auto",
        )
        self.rebuild_content_ui()

        self.editor_line_matching_vbox = VBox(
            [
                self.editor_line_matching_vbox_header,
                self.editor_line_matching_vbox_content,
                self.editor_line_matching_vbox_footer,
            ],
            layout=Layout(display="flex", max_height="calc(100vh - 200px)"),
        )

    def __init__(
        self,
        current_pgdp_page: PGDPPage | None,
        current_ocr_page: Page | None,
        monospace_font_name: str = "Courier New",
        monospace_font_path: pathlib.Path | str = None,
        page_image_change_callback: callable = None,
    ):
        self.line_matching_configuration = LineMatching.SHOW_ALL_LINES
        # self.show_exact_line_matches = False
        self._current_pgdp_page = current_pgdp_page
        self._current_ocr_page = current_ocr_page
        self.init_font(monospace_font_name, monospace_font_path)
        self.init_ui()
        self.page_image_change_callback = page_image_change_callback

        def line_change_callback():
            """
            Callback function to handle line changes.
            This function is called when a line is changed in the line editor.
            """
            logger.debug("Line change callback triggered")
            # If the line has been marked as validated, we need to update the line in the current OCR page
            self.rebuild_visible_lines()

        self.line_change_callback = line_change_callback

    def regenerate_line_editors(self):
        logger.debug("Regenerating line editors")

        self.line_editors = []
        for line in self._current_ocr_page.lines:
            logger.debug("Creating line editor for line: " + str(line.text[:20]))
            line_editor = IpynbLineEditor(
                page=self._current_ocr_page,
                pgdp_page=self._current_pgdp_page,
                line=line,
                page_image_change_callback=self.page_image_change_callback,
                line_change_callback=self.line_change_callback,
                monospace_font_name=self.monospace_font_name,
                monospace_font_path=self.monospace_font_path,
            )
            self.line_editors.append(line_editor)

        if logger.level == logging_DEBUG:
            logger.debug(lambda s: "Line editor count: " + str(len(self.line_editors)))

        logger.debug("done regenerating line editors")

    def update_line_matches(self, current_pgdp_page: PGDPPage, current_ocr_page: Page):
        self._current_pgdp_page = current_pgdp_page
        self._current_ocr_page = current_ocr_page
        self.rebuild_content_ui()

    def rebuild_visible_lines(self):
        """
        Rebuild the visible lines in the UI.
        This function is called when the line matching configuration changes.
        """
        # Clear the current content
        logger.debug("Rebuilding visible lines")

        if self._current_ocr_page is None:
            return

        logger.debug("Line count: " + str(len(self._current_ocr_page.lines)))

        boxes = []
        for idx, line in enumerate(self._current_ocr_page.lines):
            if line.ground_truth_exact_match and (
                self.line_matching_configuration == LineMatching.SHOW_ONLY_MISMATCHES
                or self.line_matching_configuration
                == LineMatching.SHOW_ONLY_UNVALIDATED_MISMATCHES
            ):
                logger.debug(
                    f"Skipping line {idx} because it is an exact match: {line.ground_truth_exact_match}"
                )
                # Skip exact matches
                continue

            if (
                self.line_matching_configuration
                == LineMatching.SHOW_ONLY_UNVALIDATED_MISMATCHES
                and line.additional_block_attributes.get("line_editor_validated", False)
            ):
                logger.debug(
                    f"Skipping line {idx} because it is validated: {line.additional_block_attributes.get('line_editor_validated', False)}"
                )
                # Skip matches that are marked as validated
                continue
            boxes.append(self.line_editors[idx].GridVBox)

        self.editor_line_matching_vbox_content.children = []
        self.editor_line_matching_vbox_content.children = boxes

    def rebuild_content_ui(self):
        logger.debug("Rebuilding content UI")
        # Clear the current content
        self.editor_line_matching_vbox_content.children = []

        if self._current_ocr_page is None:
            return

        self.regenerate_line_editors()
        self.rebuild_visible_lines()
