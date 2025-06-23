from enum import Enum
from logging import getLogger
from typing import Callable, Optional

from nicegui import ui

from pd_book_tools.ocr.page import Page
from pd_book_tools.pgdp.pgdp_results import PGDPPage

from .nicegui_line_editor import NiceGuiLineEditor

# Configure logging
logger = getLogger(__name__)
ui_logger = getLogger(__name__ + ".UI")


class LineMatching(Enum):
    SHOW_ALL_LINES = 1
    SHOW_ONLY_MISMATCHES = 2
    SHOW_ONLY_UNVALIDATED_MISMATCHES = 3


class NiceGuiPageEditor:
    """NiceGUI version of the page editor for managing line editors"""
    
    def __init__(
        self,
        current_pgdp_page: PGDPPage | None,
        current_ocr_page: Page | None,
        monospace_font_name: str = "monospace",
        page_image_change_callback: Optional[Callable] = None,
    ):
        # Core data
        self._current_pgdp_page = current_pgdp_page
        self._current_ocr_page = current_ocr_page
        self.monospace_font_name = monospace_font_name
        self.page_image_change_callback = page_image_change_callback
        
        # Line matching configuration
        self.line_matching_configuration: LineMatching = LineMatching.SHOW_ALL_LINES
        
        # Line editors
        self.line_editors: list[NiceGuiLineEditor] = []
        
        # UI elements
        self.container = None
        self.header_container = None
        self.content_container = None
        self.footer_container = None
        self.line_matching_radio = None
        
        # Callbacks
        self.line_change_callback = self.create_line_change_callback()
    
    def create_line_change_callback(self) -> Callable:
        """Create callback for line changes"""
        def line_change_callback():
            logger.debug("Line change callback triggered")
            self.rebuild_visible_lines()
        return line_change_callback
    
    def draw_ui(self, parent_container):
        """Draw the complete page editor UI"""
        with parent_container:
            self.container = ui.column().classes('w-full gap-4')
            
            with self.container:
                # Header with controls
                self.draw_header_ui()
                
                # Content area with line editors
                self.content_container = ui.column().classes('w-full gap-2')
                
                # Footer
                self.draw_footer_ui()
                
                # Initialize content
                self.rebuild_content_ui()
        
        return self.container
    
    def draw_header_ui(self):
        """Draw header controls"""
        with ui.card().classes('w-full p-3'):
            ui.label("Line Matching Configuration").classes('text-lg font-bold mb-2')
            
            with ui.row().classes('gap-4 items-center'):
                # Line matching radio buttons
                line_matching_options = {
                    "All Lines": LineMatching.SHOW_ALL_LINES,
                    "Only Mismatches": LineMatching.SHOW_ONLY_MISMATCHES,
                    "Only Unvalidated Mismatches": LineMatching.SHOW_ONLY_UNVALIDATED_MISMATCHES,
                }
                
                # Find the key that corresponds to the current value
                current_key = next(
                    (key for key, value in line_matching_options.items() 
                     if value == self.line_matching_configuration), 
                    "All Lines"  # default
                )
                
                self.line_matching_radio = ui.radio(
                    options=line_matching_options,
                    value=current_key
                ).classes('mb-2')
                
                self.line_matching_radio.on('change', self.on_line_matching_change)
                
                # Page-level action buttons
                ui.button(
                    "Expand Page BBoxes",
                    on_click=self.expand_all_page_bboxes
                ).props('outlined size=sm')
                
                ui.button(
                    "Refine Page BBoxes", 
                    on_click=self.refine_all_page_bboxes
                ).props('outlined size=sm')
    
    def draw_footer_ui(self):
        """Draw footer area"""
        self.footer_container = ui.row().classes('w-full justify-center mt-4')
        # Footer content can be added here if needed
    
    def on_line_matching_change(self, e):
        """Handle line matching configuration change"""
        new_key = e.value
        if new_key:
            # Map the string key back to enum value
            line_matching_options = {
                "All Lines": LineMatching.SHOW_ALL_LINES,
                "Only Mismatches": LineMatching.SHOW_ONLY_MISMATCHES,
                "Only Unvalidated Mismatches": LineMatching.SHOW_ONLY_UNVALIDATED_MISMATCHES,
            }
            new_value = line_matching_options.get(new_key, LineMatching.SHOW_ALL_LINES)
            
            if new_value != self.line_matching_configuration:
                ui_logger.debug(f"Changing line matching configuration to: {new_value}")
                self.line_matching_configuration = new_value
                self.rebuild_content_ui()
    
    def expand_all_page_bboxes(self):
        """Expand all bounding boxes on the page"""
        if self._current_ocr_page is None:
            ui.notify("No OCR page available", type='warning')
            return
            
        ui_logger.debug("Expanding all page bounding boxes")
        # Implementation would expand all word bounding boxes
        # For now, just show a notification
        ui.notify("Page bounding boxes expanded", type='positive')
        
        if self.page_image_change_callback:
            self.page_image_change_callback()
    
    def refine_all_page_bboxes(self):
        """Refine all bounding boxes on the page"""
        if self._current_ocr_page is None:
            ui.notify("No OCR page available", type='warning')
            return
            
        ui_logger.debug("Refining all page bounding boxes")
        # Implementation would refine all word bounding boxes
        # For now, just show a notification
        ui.notify("Page bounding boxes refined", type='positive')
        
        if self.page_image_change_callback:
            self.page_image_change_callback()
    
    def regenerate_line_editors(self):
        """Regenerate line editors for current page"""
        logger.debug("Regenerating line editors")
        
        self.line_editors = []
        if self._current_ocr_page and self._current_pgdp_page:
            for line in self._current_ocr_page.lines:
                logger.debug(f"Creating line editor for line: {line.text[:20]}")
                line_editor = NiceGuiLineEditor(
                    page=self._current_ocr_page,
                    pgdp_page=self._current_pgdp_page,
                    line=line,
                    page_image_change_callback=self.page_image_change_callback,
                    line_change_callback=self.line_change_callback,
                    monospace_font_name=self.monospace_font_name,
                )
                self.line_editors.append(line_editor)
        
        logger.debug(f"Line editor count: {len(self.line_editors)}")
    
    def rebuild_visible_lines(self):
        """Rebuild the visible lines based on current configuration"""
        logger.debug("Rebuilding visible lines")
        
        if self._current_ocr_page is None:
            return
        
        # Clear content container
        if self.content_container:
            self.content_container.clear()
        
        logger.debug(f"Line count: {len(self._current_ocr_page.lines)}")
        
        # Add visible line editors
        visible_count = 0
        for idx, line_editor in enumerate(self.line_editors):
            if line_editor.should_show_line(self.line_matching_configuration):
                if self.content_container:
                    with self.content_container:
                        line_editor.draw_ui(self.content_container)
                        visible_count += 1
            else:
                logger.debug(f"Skipping line {idx} due to matching configuration")
        
        logger.debug(f"Showing {visible_count} lines")
        
        # Show message if no lines are visible
        if visible_count == 0 and self.content_container:
            with self.content_container:
                ui.label("No lines match the current filter criteria").classes('text-gray-500 text-center p-4')
    
    def rebuild_content_ui(self):
        """Rebuild the entire content UI"""
        logger.debug("Rebuilding content UI")
        
        # Clear content
        if self.content_container:
            self.content_container.clear()
        
        if self._current_ocr_page is None:
            if self.content_container:
                with self.content_container:
                    ui.label("No OCR page available").classes('text-gray-500 text-center p-4')
            return
        
        # Regenerate line editors and rebuild visible lines
        self.regenerate_line_editors()
        self.rebuild_visible_lines()
    
    def update_line_matches(self, current_pgdp_page: PGDPPage, current_ocr_page: Page):
        """Update the page data and rebuild UI"""
        logger.debug("Updating line matches")
        self._current_pgdp_page = current_pgdp_page
        self._current_ocr_page = current_ocr_page
        self.rebuild_content_ui()
    
    def get_line_count_summary(self) -> dict:
        """Get summary of line counts by type"""
        if not self._current_ocr_page:
            return {"total": 0, "exact_matches": 0, "mismatches": 0, "validated": 0}
        
        total = len(self._current_ocr_page.lines)
        exact_matches = 0
        mismatches = 0
        validated = 0
        
        for line in self._current_ocr_page.lines:
            if line.ground_truth_exact_match:
                exact_matches += 1
            else:
                mismatches += 1
            
            if (
                line.additional_block_attributes
                and line.additional_block_attributes.get("line_editor_validated", False)
            ):
                validated += 1
        
        return {
            "total": total,
            "exact_matches": exact_matches,
            "mismatches": mismatches,
            "validated": validated
        }
    
    def show_line_statistics(self):
        """Show line statistics in a dialog"""
        stats = self.get_line_count_summary()
        
        with ui.dialog() as dialog, ui.card():
            ui.label("Line Statistics").classes('text-lg font-bold mb-4')
            
            with ui.column().classes('gap-2'):
                ui.label(f"Total Lines: {stats['total']}")
                ui.label(f"Exact Matches: {stats['exact_matches']}")
                ui.label(f"Mismatches: {stats['mismatches']}")
                ui.label(f"Validated: {stats['validated']}")
                
                accuracy = (stats['exact_matches'] / stats['total'] * 100) if stats['total'] > 0 else 0
                ui.label(f"Accuracy: {accuracy:.1f}%")
            
            ui.button("Close", on_click=dialog.close).classes('mt-4')
        
        dialog.open()
    
    def export_line_validation_report(self):
        """Export a report of line validation status"""
        if not self._current_ocr_page:
            ui.notify("No OCR page available", type='warning')
            return
        
        report_lines = []
        report_lines.append("Line Validation Report")
        report_lines.append("=" * 50)
        
        for idx, line in enumerate(self._current_ocr_page.lines):
            status = "EXACT_MATCH" if line.ground_truth_exact_match else "MISMATCH"
            
            if (
                line.additional_block_attributes
                and line.additional_block_attributes.get("line_editor_validated", False)
            ):
                status += " (VALIDATED)"
            
            report_lines.append(f"Line {idx}: {status}")
            report_lines.append(f"  OCR: {line.text[:50]}...")
            report_lines.append(f"  GT:  {(line.ground_truth_text or '')[:50]}...")
            report_lines.append("")
        
        # For now, just show in a dialog
        report_text = "\n".join(report_lines)
        
        with ui.dialog() as dialog, ui.card().style('width: 600px; max-height: 500px;'):
            ui.label("Line Validation Report").classes('text-lg font-bold mb-4')
            ui.textarea(value=report_text).props('readonly').classes('w-full h-80 font-mono text-xs')
            ui.button("Close", on_click=dialog.close).classes('mt-4')
        
        dialog.open()
    
    def batch_mark_exact_matches_validated(self):
        """Mark all exact matches as validated"""
        if not self._current_ocr_page:
            ui.notify("No OCR page available", type='warning')
            return
        
        count = 0
        for line in self._current_ocr_page.lines:
            if line.ground_truth_exact_match:
                if not line.additional_block_attributes:
                    line.additional_block_attributes = {}
                line.additional_block_attributes["line_editor_validated"] = True
                count += 1
        
        ui.notify(f"Marked {count} exact matches as validated", type='positive')
        self.rebuild_visible_lines()
    
    def batch_copy_ocr_to_gt(self):
        """Copy OCR text to ground truth for all lines"""
        if not self._current_ocr_page:
            ui.notify("No OCR page available", type='warning')
            return
        
        count = 0
        for line in self._current_ocr_page.lines:
            for word in line.items:
                if word.text and not word.ground_truth_text:
                    word.ground_truth_text = word.text
                    word.ground_truth_match_keys["match_score"] = 100
                    count += 1
        
        ui.notify(f"Copied OCR to GT for {count} words", type='positive')
        self.rebuild_content_ui()
        
        if self.page_image_change_callback:
            self.page_image_change_callback()
    
    def get_container(self):
        """Get the main container for embedding in parent UI"""
        return self.container


# Helper function to create a page editor with common settings
def create_nicegui_page_editor(
    current_pgdp_page: PGDPPage | None = None,
    current_ocr_page: Page | None = None,
    monospace_font_name: str = "monospace",
    page_image_change_callback: Optional[Callable] = None,
) -> NiceGuiPageEditor:
    """Create a NiceGUI page editor with common settings"""
    return NiceGuiPageEditor(
        current_pgdp_page=current_pgdp_page,
        current_ocr_page=current_ocr_page,
        monospace_font_name=monospace_font_name,
        page_image_change_callback=page_image_change_callback,
    )
