import base64
import cv2
from enum import Enum
from logging import getLogger
from typing import Callable, Optional

from nicegui import ui

from pd_book_tools.geometry.bounding_box import BoundingBox
from pd_book_tools.ocr.block import Block
from pd_book_tools.ocr.ground_truth_matching import update_line_with_ground_truth
from pd_book_tools.ocr.image_utilities import get_cropped_word_image
from pd_book_tools.ocr.page import Page
from pd_book_tools.pgdp.pgdp_results import PGDPPage

# Configure logging
logger = getLogger(__name__)
ui_logger = getLogger(__name__ + ".UI")


class EditorTaskType(Enum):
    NONE = 0
    SPLIT = 1
    EDITBBOX = 2


class NiceGuiLineEditor:
    """NiceGUI version of the line editor for editing individual OCR lines"""
    
    def __init__(
        self,
        page: Page,
        pgdp_page: PGDPPage,
        line: Block,
        page_image_change_callback: Optional[Callable] = None,
        line_change_callback: Optional[Callable] = None,
        monospace_font_name: str = "monospace",
    ):
        # Core data
        self._current_ocr_page = page
        self._current_pgdp_page = pgdp_page
        self._current_ocr_line = line
        self.monospace_font_name = monospace_font_name
        self.line_matches = []
        
        # Callbacks
        self.page_image_change_callback = page_image_change_callback
        self.line_change_callback = line_change_callback
        
        # Task state
        self.task_type: EditorTaskType = EditorTaskType.NONE
        self.task_match_idx: int = -1
        self.split_task_x_coordinate: int = -1
        self.split_task_word_split_idx: int = -1
        self.edit_margins = [0, 0, 0, 0]  # Left, Top, Right, Bottom
        
        # UI elements (will be set when drawing)
        self.container = None
        self.line_image = None
        self.ocr_text_display = None
        self.gt_text_display = None
        self.action_buttons_row = None
        self.word_matching_container = None
        self.task_container = None
        
        # Calculate matches and draw UI
        self.calculate_line_matches()
        
    def get_border_style(self) -> str:
        """Get border style based on line validation state"""
        if (
            self._current_ocr_line.additional_block_attributes
            and self._current_ocr_line.additional_block_attributes.get("line_editor_validated")
        ):
            return "border: 3px solid green;"
        elif self._current_ocr_line.ground_truth_exact_match:
            return "border: 3px solid gray;"
        else:
            return "border: 3px solid red;"
    
    def should_show_line(self, line_matching_config) -> bool:
        """Determine if this line should be shown based on matching configuration"""
        # Import here to avoid circular import
        from enum import Enum
        
        class LineMatching(Enum):
            SHOW_ALL_LINES = 1
            SHOW_ONLY_MISMATCHES = 2
            SHOW_ONLY_UNVALIDATED_MISMATCHES = 3
        
        if self._current_ocr_line.ground_truth_exact_match and (
            line_matching_config == LineMatching.SHOW_ONLY_MISMATCHES
            or line_matching_config == LineMatching.SHOW_ONLY_UNVALIDATED_MISMATCHES
        ):
            return False
            
        if not self._current_ocr_line.additional_block_attributes:
            self._current_ocr_line.additional_block_attributes = {}
            
        if (
            line_matching_config == LineMatching.SHOW_ONLY_UNVALIDATED_MISMATCHES
            and self._current_ocr_line.additional_block_attributes.get("line_editor_validated", False)
        ):
            return False
            
        return len(self.line_matches) > 0
    
    def draw_ui(self, parent_container):
        """Draw the complete line editor UI"""
        if not self.line_matches:
            return None
            
        with parent_container:
            with ui.card().style(f"margin: 5px; padding: 5px; {self.get_border_style()}"):
                self.container = ui.column().classes('w-full gap-2')
                
                with self.container:
                    # Line image
                    self.draw_line_image()
                    
                    # OCR and GT text displays
                    self.draw_text_displays()
                    
                    # Action buttons
                    self.draw_action_buttons()
                    
                    # Word matching table
                    self.draw_word_matching_table()
                    
                    # Active task UI
                    self.task_container = ui.column()
                    self.draw_active_task()
                    
        return self.container
    
    def draw_line_image(self):
        """Draw the cropped line image"""
        if (
            self._current_ocr_line.bounding_box
            and self._current_ocr_page.cv2_numpy_page_image is not None
        ):
            # Get cropped line image
            bbox = self._current_ocr_line.bounding_box
            h, w = self._current_ocr_page.cv2_numpy_page_image.shape[:2]
            scaled_bbox = bbox.scale(w, h)
            
            x1, y1, x2, y2 = int(scaled_bbox.minX), int(scaled_bbox.minY), int(scaled_bbox.maxX), int(scaled_bbox.maxY)
            cropped_img = self._current_ocr_page.cv2_numpy_page_image[y1:y2, x1:x2]
            
            # Encode as base64 for display
            _, buffer = cv2.imencode('.png', cropped_img)
            img_b64 = base64.b64encode(buffer).decode()
            
            self.line_image = ui.image(f"data:image/png;base64,{img_b64}").classes('max-h-16')
        else:
            ui.html("<span style='color: red;'>Error: No bounding box for line</span>")
    
    def draw_text_displays(self):
        """Draw OCR and ground truth text displays"""
        with ui.row().classes('w-full gap-4'):
            with ui.column().classes('flex-1'):
                ui.label("OCR Text:").classes('text-sm font-bold')
                color = "lightgray" if self._current_ocr_line.ground_truth_exact_match else "black"
                self.ocr_text_display = ui.html(
                    f"<span style='font-family: {self.monospace_font_name}; color: {color}; font-size: 14px;'>"
                    f"{self._current_ocr_line.text}</span>"
                ).classes('border p-2')
            
            with ui.column().classes('flex-1'):
                ui.label("Ground Truth:").classes('text-sm font-bold')
                color = "lightgray" if self._current_ocr_line.ground_truth_exact_match else "black"
                gt_text = self._current_ocr_line.ground_truth_text or "&nbsp;"
                self.gt_text_display = ui.html(
                    f"<span style='font-family: {self.monospace_font_name}; color: {color}; font-size: 14px;'>"
                    f"{gt_text}</span>"
                ).classes('border p-2')
    
    def draw_action_buttons(self):
        """Draw line action buttons"""
        with ui.row().classes('gap-2'):
            ui.button("Copy Line to GT", on_click=self.copy_ocr_to_gt).props('size=sm')
            ui.button("Delete Line", on_click=self.delete_line).props('size=sm color=negative')
            ui.button("Mark as Validated", on_click=self.mark_validated).props('size=sm color=positive')
            
            ui.label("Crop:").classes('self-center')
            ui.button("T", on_click=lambda: self.crop_words('T')).props('size=sm').style('width: 30px')
            ui.button("B", on_click=lambda: self.crop_words('B')).props('size=sm').style('width: 30px')
            ui.button("A", on_click=lambda: self.crop_words('A')).props('size=sm').style('width: 30px')
    
    def draw_word_matching_table(self):
        """Draw the word matching table with individual word controls"""
        if not self.line_matches:
            return
            
        ui.label("Word Matching:").classes('text-sm font-bold mt-4')
        
        with ui.row().classes('w-full gap-2 flex-wrap'):
            for match in self.line_matches:
                self.draw_word_match_card(match)
    
    def draw_word_match_card(self, match):
        """Draw a card for individual word matching"""
        with ui.card().classes('p-2').style('min-width: 120px; border: 1px solid #ccc;'):
            with ui.column().classes('gap-1'):
                # Word image
                if match.get("img_tag_text"):
                    ui.html(match["img_tag_text"]).classes('text-center')
                else:
                    ui.label("No Image").classes('text-center text-xs text-gray-500')
                
                # OCR text
                ocr_color = match.get("ocr_text_color", "black")
                ocr_text = match.get("ocr_text", "No OCR")
                ui.html(
                    f"<span style='font-family: {self.monospace_font_name}; color: {ocr_color}; font-size: 12px;'>"
                    f"{ocr_text}</span>"
                ).classes('text-center border-b pb-1')
                
                # Ground truth text (editable)
                gt_text = match.get("gt_text", "")
                gt_input = ui.input(value=gt_text, placeholder="Ground truth").classes('text-xs')
                gt_input.on('blur', lambda e, m=match: self.update_gt_text(e.sender.value, m) if hasattr(e.sender, 'value') else None)
                
                # Action buttons
                if self.task_type == EditorTaskType.NONE:
                    with ui.row().classes('gap-1 justify-center'):
                        ui.button("X", on_click=lambda _=None, m=match: self.delete_match(m)).props('size=xs color=negative').style('width: 20px')
                        
                        word = match.get("word")
                        if word:
                            if match.get("word_idx", 0) > 0:
                                ui.button("ML", on_click=lambda _=None, m=match: self.merge_left(m)).props('size=xs').style('width: 25px')
                            
                            if match.get("word_idx", 0) < len(self._current_ocr_line.items) - 1:
                                ui.button("MR", on_click=lambda _=None, m=match: self.merge_right(m)).props('size=xs').style('width: 25px')
                            
                            ui.button("EB", on_click=lambda _=None, m=match: self.start_edit_bbox_task(m)).props('size=xs').style('width: 25px')
                            ui.button("SP", on_click=lambda _=None, m=match: self.start_split_task(m)).props('size=xs').style('width: 25px')
                        
                        # Crop buttons
                        if word:
                            with ui.row().classes('gap-1 mt-1'):
                                ui.button("CT", on_click=lambda _=None, m=match: self.crop_word_top(m)).props('size=xs').style('width: 25px')
                                ui.button("CB", on_click=lambda _=None, m=match: self.crop_word_bottom(m)).props('size=xs').style('width: 25px')
                                ui.button("CA", on_click=lambda _=None, m=match: self.crop_word_both(m)).props('size=xs').style('width: 25px')
    
    def draw_active_task(self):
        """Draw UI for active tasks (split or edit bbox)"""
        if self.task_container is None:
            return
            
        self.task_container.clear()
        
        with self.task_container:
            if self.task_type == EditorTaskType.SPLIT:
                self.draw_split_task_ui()
            elif self.task_type == EditorTaskType.EDITBBOX:
                self.draw_edit_bbox_task_ui()
    
    def draw_split_task_ui(self):
        """Draw split task interface"""
        ui.label("Split Task Active").classes('text-lg font-bold text-blue-600')
        
        # Split image with line
        match = self.line_matches[self.task_match_idx]
        img_ndarray = match["img_ndarray"]
        h, w = img_ndarray.shape[:2]
        
        if self.split_task_x_coordinate == -1:
            self.split_task_x_coordinate = int(w / 2)
        
        # Draw split line on image
        split_img = cv2.line(
            img=img_ndarray.copy(),
            pt1=(self.split_task_x_coordinate, 0),
            pt2=(self.split_task_x_coordinate, h),
            color=(0, 0, 255),  # Red line
            thickness=1,
        )
        
        _, buffer = cv2.imencode('.png', split_img)
        img_b64 = base64.b64encode(buffer).decode()
        ui.image(f"data:image/png;base64,{img_b64}").classes('max-h-20')
        
        # Split controls
        with ui.row().classes('gap-2 mt-2'):
            ui.button("Cancel", on_click=self.cancel_split_task).props('color=negative size=sm')
            ui.button("<1p", on_click=lambda: self.split_move_pixels(-1)).props('size=sm').style('width: 35px')
            ui.button(">1p", on_click=lambda: self.split_move_pixels(1)).props('size=sm').style('width: 35px')
            ui.button("<5%", on_click=lambda: self.split_move_percent(-0.05)).props('size=sm').style('width: 35px')
            ui.button(">5%", on_click=lambda: self.split_move_percent(0.05)).props('size=sm').style('width: 35px')
            ui.button("<20%", on_click=lambda: self.split_move_percent(-0.2)).props('size=sm').style('width: 40px')
            ui.button(">20%", on_click=lambda: self.split_move_percent(0.2)).props('size=sm').style('width: 40px')
            ui.button("Execute Split", on_click=self.execute_split).props('color=positive size=sm')
        
        # Text split controls
        ui.label("Text Split Position:").classes('mt-2')
        ocr_text = match.get("ocr_text", "")
        if self.split_task_word_split_idx == -1:
            self.split_task_word_split_idx = len(ocr_text) // 2
        
        # Show text with split indicator
        split_text = (
            ocr_text[:self.split_task_word_split_idx] + 
            "|" + 
            ocr_text[self.split_task_word_split_idx:]
        )
        ui.html(f"<span style='font-family: {self.monospace_font_name}; font-size: 16px;'>{split_text}</span>")
        
        with ui.row().classes('gap-2 mt-1'):
            ui.button("<", on_click=lambda: self.split_move_text(-1)).props('size=sm').style('width: 25px')
            ui.button(">", on_click=lambda: self.split_move_text(1)).props('size=sm').style('width: 25px')
    
    def draw_edit_bbox_task_ui(self):
        """Draw edit bounding box task interface"""
        ui.label("Edit Bounding Box Task Active").classes('text-lg font-bold text-blue-600')
        
        # Show current bbox image with modifications
        self.update_edit_bbox_display()
        
        with ui.row().classes('gap-2 mt-2'):
            ui.button("Cancel", on_click=self.cancel_edit_bbox_task).props('color=negative size=sm')
            ui.button("Refine", on_click=self.edit_bbox_refine).props('size=sm')
            ui.button("Save", on_click=self.execute_edit_bbox_task).props('color=positive size=sm')
        
        # Margin adjustment controls
        ui.label("Adjust Margins:").classes('mt-2')
        
        with ui.grid(columns=3).classes('gap-2 w-fit'):
            # Top row
            ui.label("")  # Empty cell
            with ui.column().classes('items-center'):
                ui.label("Top").classes('text-xs')
                with ui.row().classes('gap-1'):
                    ui.button("↑1", on_click=lambda: self.edit_bbox_adjust_margin("T", -1)).props('size=xs')
                    ui.button("↑5", on_click=lambda: self.edit_bbox_adjust_margin("T", -5)).props('size=xs')
                    ui.button("↑10", on_click=lambda: self.edit_bbox_adjust_margin("T", -10)).props('size=xs')
                with ui.row().classes('gap-1'):
                    ui.button("↓1", on_click=lambda: self.edit_bbox_adjust_margin("T", 1)).props('size=xs')
                    ui.button("↓5", on_click=lambda: self.edit_bbox_adjust_margin("T", 5)).props('size=xs')
                    ui.button("↓10", on_click=lambda: self.edit_bbox_adjust_margin("T", 10)).props('size=xs')
            ui.label("")  # Empty cell
            
            # Middle row
            with ui.column().classes('items-center'):
                ui.label("Left").classes('text-xs')
                with ui.row().classes('gap-1'):
                    ui.button("←1", on_click=lambda: self.edit_bbox_adjust_margin("L", -1)).props('size=xs')
                    ui.button("←5", on_click=lambda: self.edit_bbox_adjust_margin("L", -5)).props('size=xs')
                    ui.button("←10", on_click=lambda: self.edit_bbox_adjust_margin("L", -10)).props('size=xs')
                with ui.row().classes('gap-1'):
                    ui.button("→1", on_click=lambda: self.edit_bbox_adjust_margin("L", 1)).props('size=xs')
                    ui.button("→5", on_click=lambda: self.edit_bbox_adjust_margin("L", 5)).props('size=xs')
                    ui.button("→10", on_click=lambda: self.edit_bbox_adjust_margin("L", 10)).props('size=xs')
            
            ui.label("Image").classes('text-center self-center')  # Center cell
            
            with ui.column().classes('items-center'):
                ui.label("Right").classes('text-xs')
                with ui.row().classes('gap-1'):
                    ui.button("←1", on_click=lambda: self.edit_bbox_adjust_margin("R", -1)).props('size=xs')
                    ui.button("←5", on_click=lambda: self.edit_bbox_adjust_margin("R", -5)).props('size=xs')
                    ui.button("←10", on_click=lambda: self.edit_bbox_adjust_margin("R", -10)).props('size=xs')
                with ui.row().classes('gap-1'):
                    ui.button("→1", on_click=lambda: self.edit_bbox_adjust_margin("R", 1)).props('size=xs')
                    ui.button("→5", on_click=lambda: self.edit_bbox_adjust_margin("R", 5)).props('size=xs')
                    ui.button("→10", on_click=lambda: self.edit_bbox_adjust_margin("R", 10)).props('size=xs')
            
            # Bottom row
            ui.label("")  # Empty cell
            with ui.column().classes('items-center'):
                ui.label("Bottom").classes('text-xs')
                with ui.row().classes('gap-1'):
                    ui.button("↑1", on_click=lambda: self.edit_bbox_adjust_margin("B", -1)).props('size=xs')
                    ui.button("↑5", on_click=lambda: self.edit_bbox_adjust_margin("B", -5)).props('size=xs')
                    ui.button("↑10", on_click=lambda: self.edit_bbox_adjust_margin("B", -10)).props('size=xs')
                with ui.row().classes('gap-1'):
                    ui.button("↓1", on_click=lambda: self.edit_bbox_adjust_margin("B", 1)).props('size=xs')
                    ui.button("↓5", on_click=lambda: self.edit_bbox_adjust_margin("B", 5)).props('size=xs')
                    ui.button("↓10", on_click=lambda: self.edit_bbox_adjust_margin("B", 10)).props('size=xs')
            ui.label("")  # Empty cell
    
    def update_edit_bbox_display(self):
        """Update the edit bbox image display"""
        # This would show the current bounding box with margin adjustments
        # Implementation would involve getting the modified bbox and displaying it
        pass
    
    # Event handlers and core functionality
    
    def copy_ocr_to_gt(self):
        """Copy OCR text to ground truth for all words in line"""
        for word in self._current_ocr_line.items:
            word.ground_truth_text = word.text
            word.ground_truth_match_keys["match_score"] = word.fuzz_score_against(word.ground_truth_text)
        self.refresh_ui()
        ui.notify("Copied OCR text to ground truth", type='positive')
    
    def delete_line(self):
        """Delete the current line"""
        # This would need to be handled by the parent page editor
        ui.notify("Line deletion requested", type='info')
        if self.line_change_callback:
            self.line_change_callback()
    
    def mark_validated(self):
        """Mark line as validated"""
        if not self._current_ocr_line.additional_block_attributes:
            self._current_ocr_line.additional_block_attributes = {}
        self._current_ocr_line.additional_block_attributes["line_editor_validated"] = True
        self.refresh_ui()
        ui.notify("Line marked as validated", type='positive')
        if self.line_change_callback:
            self.line_change_callback()
    
    def crop_words(self, crop_type: str):
        """Crop words based on type (T=top, B=bottom, A=all)"""
        if self._current_ocr_page.cv2_numpy_page_image is None:
            ui.notify("No page image available for cropping", type='negative')
            return
            
        img_ndarray = self._current_ocr_page.cv2_numpy_page_image
        
        for match in self.line_matches:
            word = match.get("word")
            if not word:
                continue
                
            if crop_type == 'T':
                word.crop_top(img_ndarray)
            elif crop_type == 'B':
                word.crop_bottom(img_ndarray)
            elif crop_type == 'A':
                word.crop_top(img_ndarray)
                word.crop_bottom(img_ndarray)
        
        self._current_ocr_line.recompute_bounding_box()
        self._current_ocr_page.recompute_bounding_box()
        self.refresh_ui()
        ui.notify(f"Cropped words ({crop_type})", type='positive')
        
        if self.page_image_change_callback:
            self.page_image_change_callback()
    
    def update_gt_text(self, new_text: str, match: dict):
        """Update ground truth text for a word"""
        word = match.get("word")
        if word:
            word.ground_truth_text = new_text
            word.ground_truth_match_keys["match_score"] = word.fuzz_score_against(new_text)
            match["gt_text"] = new_text
            if self.page_image_change_callback:
                self.page_image_change_callback()
    
    def delete_match(self, match: dict):
        """Delete a word match"""
        word = match.get("word")
        if word:
            self._current_ocr_line.remove_item(word)
            self._current_ocr_page.remove_empty_items()
        else:
            # Handle unmatched ground truth deletion
            if self._current_ocr_line.unmatched_ground_truth_words:
                self._current_ocr_line.unmatched_ground_truth_words = [
                    unmatched_gt_word
                    for unmatched_gt_word in self._current_ocr_line.unmatched_ground_truth_words
                    if unmatched_gt_word[0] != match.get("word_idx")
                    and unmatched_gt_word[1] != match.get("gt_text")
                ]
        
        self.refresh_ui()
        ui.notify("Match deleted", type='info')
        if self.page_image_change_callback:
            self.page_image_change_callback()
    
    def merge_left(self, match: dict):
        """Merge word with previous word"""
        match_idx = self.line_matches.index(match)
        if match_idx <= 0:
            ui.notify("No previous word to merge with", type='warning')
            return
            
        prev_match = self.line_matches[match_idx - 1]
        word = match.get("word")
        prev_word = prev_match.get("word")
        
        if word and prev_word:
            prev_word.merge(word)
            self._current_ocr_line.remove_item(word)
            ui.notify("Words merged", type='positive')
        
        self.refresh_ui()
        if self.page_image_change_callback:
            self.page_image_change_callback()
    
    def merge_right(self, match: dict):
        """Merge word with next word"""
        word_idx = match.get("word_idx", 0)
        if word_idx >= len(self._current_ocr_line.items) - 1:
            ui.notify("No next word to merge with", type='warning')
            return
            
        word = match.get("word")
        next_word = self._current_ocr_line.items[word_idx + 1]
        
        if word and next_word:
            word.merge(next_word)
            self._current_ocr_line.remove_item(next_word)
            ui.notify("Words merged", type='positive')
        
        self.refresh_ui()
        if self.page_image_change_callback:
            self.page_image_change_callback()
    
    def crop_word_top(self, match: dict):
        """Crop top of word bounding box"""
        word = match.get("word")
        if not word or self._current_ocr_page.cv2_numpy_page_image is None:
            return
            
        word.crop_top(self._current_ocr_page.cv2_numpy_page_image)
        self._current_ocr_line.recompute_bounding_box()
        self._current_ocr_page.recompute_bounding_box()
        self.refresh_ui()
        ui.notify("Word top cropped", type='positive')
        
        if self.page_image_change_callback:
            self.page_image_change_callback()
    
    def crop_word_bottom(self, match: dict):
        """Crop bottom of word bounding box"""
        word = match.get("word")
        if not word or self._current_ocr_page.cv2_numpy_page_image is None:
            return
            
        word.crop_bottom(self._current_ocr_page.cv2_numpy_page_image)
        self._current_ocr_line.recompute_bounding_box()
        self._current_ocr_page.recompute_bounding_box()
        self.refresh_ui()
        ui.notify("Word bottom cropped", type='positive')
        
        if self.page_image_change_callback:
            self.page_image_change_callback()
    
    def crop_word_both(self, match: dict):
        """Crop both top and bottom of word bounding box"""
        word = match.get("word")
        if not word or self._current_ocr_page.cv2_numpy_page_image is None:
            return
            
        word.crop_top(self._current_ocr_page.cv2_numpy_page_image)
        word.crop_bottom(self._current_ocr_page.cv2_numpy_page_image)
        self._current_ocr_line.recompute_bounding_box()
        self._current_ocr_page.recompute_bounding_box()
        self.refresh_ui()
        ui.notify("Word cropped (top and bottom)", type='positive')
        
        if self.page_image_change_callback:
            self.page_image_change_callback()
    
    # Task management
    
    def start_split_task(self, match: dict):
        """Start split task for a word"""
        self.task_type = EditorTaskType.SPLIT
        self.task_match_idx = match["idx"]
        self.split_task_x_coordinate = -1
        self.split_task_word_split_idx = -1
        self.draw_active_task()
        ui.notify("Split task started", type='info')
    
    def cancel_split_task(self):
        """Cancel split task"""
        self.task_type = EditorTaskType.NONE
        self.task_match_idx = -1
        self.split_task_x_coordinate = -1
        self.split_task_word_split_idx = -1
        self.draw_active_task()
        ui.notify("Split task cancelled", type='info')
    
    def start_edit_bbox_task(self, match: dict):
        """Start edit bounding box task"""
        self.task_type = EditorTaskType.EDITBBOX
        self.task_match_idx = match["idx"]
        self.edit_margins = [0, 0, 0, 0]
        self.draw_active_task()
        ui.notify("Edit bbox task started", type='info')
    
    def cancel_edit_bbox_task(self):
        """Cancel edit bounding box task"""
        self.task_type = EditorTaskType.NONE
        self.task_match_idx = -1
        self.edit_margins = [0, 0, 0, 0]
        self.draw_active_task()
        ui.notify("Edit bbox task cancelled", type='info')
    
    def split_move_pixels(self, amount: int):
        """Move split line by pixel amount"""
        if self.task_match_idx < 0:
            return
            
        match = self.line_matches[self.task_match_idx]
        w = match["img_ndarray"].shape[1]
        self.split_task_x_coordinate = max(0, min(w, self.split_task_x_coordinate + amount))
        self.draw_active_task()
    
    def split_move_percent(self, amount: float):
        """Move split line by percentage amount"""
        if self.task_match_idx < 0:
            return
            
        match = self.line_matches[self.task_match_idx]
        w = match["img_ndarray"].shape[1]
        pixel_amount = int(w * amount)
        self.split_task_x_coordinate = max(0, min(w, self.split_task_x_coordinate + pixel_amount))
        self.draw_active_task()
    
    def split_move_text(self, amount: int):
        """Move text split position by character amount"""
        if self.task_match_idx < 0:
            return
            
        match = self.line_matches[self.task_match_idx]
        ocr_text = match.get("ocr_text", "")
        self.split_task_word_split_idx = max(0, min(len(ocr_text), self.split_task_word_split_idx + amount))
        self.draw_active_task()
    
    def execute_split(self):
        """Execute the split operation"""
        if self.task_match_idx < 0:
            return
            
        match = self.line_matches[self.task_match_idx]
        split_word_idx = match.get("word_idx", 0)
        
        # Convert pixel coordinate to normalized
        w = match["img_ndarray"].shape[1]
        normalized_split_x_offset = float(self.split_task_x_coordinate) / float(w)
        
        # Perform the split
        self._current_ocr_line.split_word(
            split_word_index=split_word_idx,
            bbox_split_offset=normalized_split_x_offset,
            character_split_index=self.split_task_word_split_idx,
        )
        
        # Expand and refine bounding boxes
        for word in self._current_ocr_line.items:
            word.bounding_box = word.bounding_box.expand_to_content(
                image=self._current_ocr_page.cv2_numpy_page_image
            )
        
        self._current_ocr_line.refine_bounding_boxes(
            image=self._current_ocr_page.cv2_numpy_page_image, padding_px=1
        )
        
        # Rematch with ground truth
        ocr_line_tuple = tuple([w.text for w in self._current_ocr_line.items])
        if self._current_ocr_line.base_ground_truth_text:
            ground_truth_tuple = tuple(self._current_ocr_line.base_ground_truth_text.split(" "))
        else:
            ground_truth_tuple = tuple([])
        
        update_line_with_ground_truth(
            self._current_ocr_line,
            ocr_line_tuple=ocr_line_tuple,
            ground_truth_tuple=ground_truth_tuple,
        )
        
        # Clean up task
        self.task_type = EditorTaskType.NONE
        self.task_match_idx = -1
        self.split_task_x_coordinate = -1
        self.split_task_word_split_idx = -1
        
        self.refresh_ui()
        ui.notify("Word split executed successfully", type='positive')
        
        if self.page_image_change_callback:
            self.page_image_change_callback()
    
    def edit_bbox_adjust_margin(self, margin: str, amount: int):
        """Adjust bounding box margins"""
        margin_map = {"L": 0, "T": 1, "R": 2, "B": 3}
        if margin in margin_map:
            self.edit_margins[margin_map[margin]] += amount
            self.update_edit_bbox_display()
    
    def edit_bbox_refine(self):
        """Refine the bounding box"""
        if self.task_match_idx < 0:
            return
            
        match = self.line_matches[self.task_match_idx]
        word = match.get("word")
        if word and self._current_ocr_page.cv2_numpy_page_image is not None:
            word.bounding_box = word.bounding_box.expand_to_content(
                image=self._current_ocr_page.cv2_numpy_page_image
            )
            self.update_edit_bbox_display()
            ui.notify("Bounding box refined", type='positive')
    
    def execute_edit_bbox_task(self):
        """Execute bounding box edit"""
        if self.task_match_idx < 0:
            return
            
        match = self.line_matches[self.task_match_idx]
        word = match.get("word")
        
        if not word or self._current_ocr_page.cv2_numpy_page_image is None:
            ui.notify("Cannot execute bbox edit", type='negative')
            return
        
        # Apply margin adjustments
        h, w = self._current_ocr_page.cv2_numpy_page_image.shape[:2]
        word_bbox = word.bounding_box.scale(w, h)
        
        minX = max(0, min(w, word_bbox.minX + self.edit_margins[0]))
        minY = max(0, min(h, word_bbox.minY + self.edit_margins[1]))
        maxX = max(0, min(w, word_bbox.maxX + self.edit_margins[2]))
        maxY = max(0, min(h, word_bbox.maxY + self.edit_margins[3]))
        
        modified_bbox = BoundingBox.from_ltrb(minX, minY, maxX, maxY)
        word.bounding_box = modified_bbox.normalize(w, h)
        
        # Refine the bounding box
        self._current_ocr_line.refine_bounding_boxes(
            image=self._current_ocr_page.cv2_numpy_page_image, padding_px=1
        )
        
        # Clean up task
        self.task_type = EditorTaskType.NONE
        self.task_match_idx = -1
        self.edit_margins = [0, 0, 0, 0]
        
        self.refresh_ui()
        ui.notify("Bounding box updated successfully", type='positive')
        
        if self.page_image_change_callback:
            self.page_image_change_callback()
    
    def calculate_line_matches(self):
        """Calculate word matches for the line"""
        if self._current_ocr_page.cv2_numpy_page_image is None:
            self.line_matches = []
            return
        
        matches = []
        
        for word_idx, word in enumerate(self._current_ocr_line.items):
            gt_text = word.ground_truth_text or ""
            ocr_text = word.text or ""
            
            # Get word image
            try:
                img_ndarray, _, _, data_src_string = get_cropped_word_image(
                    img=self._current_ocr_page.cv2_numpy_page_image,
                    word=word,
                )
                
                img_tag_text = f'<img src="data:image/png;base64,{data_src_string}" style="height: 14px;" />'
            except Exception:
                img_ndarray = None
                data_src_string = None
                img_tag_text = None
            
            # Determine colors based on match quality
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
        
        # Add unmatched ground truth words
        if self._current_ocr_line.unmatched_ground_truth_words:
            for unmatched_words in reversed(self._current_ocr_line.unmatched_ground_truth_words):
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
        
        # Add indices
        for idx, match in enumerate(matches):
            match["idx"] = idx
        
        self.line_matches = matches
    
    def refresh_ui(self):
        """Refresh the UI display"""
        self.calculate_line_matches()
        if self.container:
            # This would require rebuilding the UI
            # For now, just update what we can
            pass
