import os
import io
import sys
import base64
import json
import time
import uuid
import math
import asyncio
import shutil
from math import ceil, sqrt
from typing import List, Union
from pathlib import Path

import fal_client
import requests
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QFileDialog, QSplitter,
    QScrollArea, QFrame, QProgressBar, QMessageBox, QDialog, QInputDialog, QGridLayout,
    QGraphicsOpacityEffect
)
from PyQt6.QtCore import (
    Qt, pyqtSignal, QUrl, QTimer, QThread, QSize, QRectF, QPropertyAnimation, QStandardPaths
)
from PyQt6.QtGui import (
    QPixmap, QImage, QColor, QPalette, QFont, QIcon, QTextCursor,
    QTextCharFormat, QDesktopServices, QPainter, QPen, QPainterPath
)
from PyQt6.QtWebEngineWidgets import QWebEngineView

# -------------------------------------------------------------------
# Helper to create a custom dark-grey file icon
def get_dark_grey_file_icon(size=24):
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    dark_grey = QColor("#555555")
    pen = QPen(dark_grey)
    pen.setWidth(2)
    painter.setPen(pen)
    painter.setBrush(dark_grey)
    fold_size = size // 3
    path = QPainterPath()
    path.moveTo(0, 0)
    path.lineTo(size - fold_size, 0)
    path.lineTo(size, fold_size)
    path.lineTo(size, size)
    path.lineTo(0, size)
    path.closeSubpath()
    painter.drawPath(path)
    painter.end()
    return QIcon(pixmap)

# -------------------------------------------------------------------
# Helper to create a white download icon
def get_download_icon(size=24):
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(Qt.GlobalColor.white)
    pen.setWidth(2)
    painter.setPen(pen)
    painter.drawLine(size//2, 4, size//2, size-8)
    painter.drawLine(size//2, size-8, size//2 - 4, size-12)
    painter.drawLine(size//2, size-8, size//2 + 4, size-12)
    painter.end()
    return QIcon(pixmap)

# -------------------------------------------------------------------
# A clickable label that emits a signal when clicked.
class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        self.clicked.emit()

# -------------------------------------------------------------------
# Popup widget to show enlarged image with vision description and a download button.
class ImagePopup(QWidget):
    def __init__(self, image_path: str, vision_text: str, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.vision_text = vision_text
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Popup)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.init_ui()
        self.fade_in()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        # Left: enlarged image container
        image_container = QWidget()
        image_container.setFixedSize(400, 400)
        image_container.setStyleSheet("background-color: #000000; border: 1px solid #666666;")
        ic_layout = QVBoxLayout(image_container)
        ic_layout.setContentsMargins(0, 0, 0, 0)
        self.image_label = ClickableLabel()
        pixmap = QPixmap(self.image_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ic_layout.addWidget(self.image_label)
        # Overlay download button on image
        self.download_btn = QPushButton("", image_container)
        self.download_btn.setIcon(get_download_icon())
        self.download_btn.setIconSize(QSize(24, 24))
        self.download_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0,0,0,0.5);
                border: none;
            }
            QPushButton:hover {
                background-color: rgba(0,0,0,0.7);
            }
        """)
        self.download_btn.setFixedSize(30, 30)
        self.download_btn.move(image_container.width() - self.download_btn.width() - 5, 5)
        self.download_btn.clicked.connect(self.download_image)
        # Right: vision description
        vision_label = QLabel(self.vision_text if self.vision_text else "No vision description available.")
        vision_label.setWordWrap(True)
        vision_label.setStyleSheet("font-size: 16px; color: #f0f0f0;")
        vision_label.setFixedWidth(200)
        layout.addWidget(image_container)
        layout.addWidget(vision_label)

    def fade_in(self):
        self.effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.effect)
        self.anim = QPropertyAnimation(self.effect, b"opacity")
        self.anim.setDuration(300)
        self.anim.setStartValue(0)
        self.anim.setEndValue(1)
        self.anim.start()

    def download_image(self):
        default_dir = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DownloadLocation)
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", os.path.join(default_dir, os.path.basename(self.image_path)), "Image Files (*.png *.jpg *.jpeg *.webp)")
        if save_path:
            try:
                shutil.copy(self.image_path, save_path)
            except Exception as e:
                QMessageBox.critical(self, "Download Error", str(e))

# -------------------------------------------------------------------
# TYPEWRITER THREAD FOR PLACEHOLDER ANIMATION
class TypewriterThread(QThread):
    update_text = pyqtSignal(str)
    
    def __init__(self, prompts: List[str]):
        super().__init__()
        self.prompts = prompts
        self.running = True
        
    def run(self):
        while self.running:
            for prompt in self.prompts:
                for i in range(1, len(prompt) + 1):
                    if not self.running:
                        return
                    self.update_text.emit(prompt[:i])
                    time.sleep(0.05)
                time.sleep(2)
                for i in range(len(prompt), 0, -1):
                    if not self.running:
                        return
                    self.update_text.emit(prompt[:i])
                    time.sleep(0.025)
                time.sleep(0.5)
    
    def stop(self):
        self.running = False

# -------------------------------------------------------------------
# TEXT REFINEMENT WORKER (Uses Gemini 2.0 Flash for text refinement)
class TextRefinementWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, conversation_history: List[dict], current_request: str):
        super().__init__()
        self.conversation_history = conversation_history
        self.current_request = current_request
        
    def run(self):
        try:
            sorted_history = sorted(self.conversation_history[-30:], key=lambda msg: msg.get("timestamp", 0))
            history_text = ""
            for msg in sorted_history:
                role = msg["role"]
                content = msg.get("text", "")
                if msg.get("images"):
                    image_info_list = []
                    for img_path in msg["images"]:
                        img_name = os.path.basename(img_path)
                        vision_info = msg.get("vision", "No vision description")
                        image_info_list.append(f"{img_name} (vision: {vision_info})")
                    content += " [Images: " + ", ".join(image_info_list) + "]"
                history_text += f"{role}: {content}\n"
            refined_prompt = (
                "Based on the conversation history below, determine the final image generation request WITHOUT changing the meaning of user request, but just refining the grammar. "
                "Consider the image details provided so that you can decide which images to send to the Gemini 2.0 flash experimental image generation model.\n"
                f"Conversation history:\n{history_text}\n"
                f"User request: {self.current_request}\nFinal prompt:"
            )
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=refined_prompt,
                config=types.GenerateContentConfig(response_modalities=["Text"])
            )
            final = response.text.strip()
            self.finished.emit(final)
        except Exception as e:
            self.error.emit(str(e))

# -------------------------------------------------------------------
# IMAGE GENERATION WORKER (Modified to accept multiple images)
class ImageGenerationWorker(QThread):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, refined_prompt: str, image_paths: List[str] = None):
        super().__init__()
        self.refined_prompt = refined_prompt
        self.image_paths = image_paths if image_paths is not None else []
        
    def run(self):
        try:
            self.progress.emit(20)
            if self.image_paths:
                images = []
                for path in self.image_paths:
                    with Image.open(path) as img:
                        images.append(img.copy())
                contents = [self.refined_prompt] + images
            else:
                contents = self.refined_prompt
            self.progress.emit(40)
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp-image-generation",
                contents=contents,
                config=types.GenerateContentConfig(response_modalities=["Text", "Image"])
            )
            self.progress.emit(80)
            result = {
                "text": response.text or "Generated image",
                "images": []
            }
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    result["images"].append(part.inline_data.data)
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

# -------------------------------------------------------------------
# ASYNCHRONOUS FAL.AI 3D CONVERSION WORKER
class FalWorker(QThread):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    log_message = pyqtSignal(str)
    
    def __init__(self, image_url):
        super().__init__()
        self.image_url = image_url
        
    def run(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.async_run())
            loop.close()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"Error in async 3D conversion: {str(e)}")
    
    async def async_run(self):
        self.progress.emit(10)
        self.log_message.emit("FalWorker: Starting async 3D conversion...")
        handler = await fal_client.submit_async(
            "fal-ai/trellis",
            arguments={
                "image_url": self.image_url,
                "texture_size": 1024
            }
        )
        async for event in handler.iter_events(with_logs=True):
            self.log_message.emit(f"FalWorker event: {event}")
            self.progress.emit(50)
        result = await handler.get()
        self.progress.emit(90)
        self.log_message.emit("FalWorker: 3D model generation complete!")
        processed_result = {
            "model_url": result["model_mesh"]["url"],
            "content_type": result["model_mesh"]["content_type"],
            "file_name": result["model_mesh"]["file_name"],
            "file_size": result["model_mesh"]["file_size"]
        }
        self.progress.emit(100)
        return processed_result

# -------------------------------------------------------------------
# VISION WORKER (Uses Gemini 2.0 Flash Experimental Vision Model)
class VisionWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, image_paths: List[str]):
        super().__init__()
        self.image_paths = image_paths
        
    def run(self):
        try:
            images = []
            for path in self.image_paths:
                with Image.open(path) as img:
                    images.append(img.copy())
            prompt = ("Can you describe in details the image? Provide details on colors, texts, main objects, styles, and scene context. Write a short phrase of max 3 lines.")
            contents = [prompt] + images
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=contents,
                config=types.GenerateContentConfig(response_modalities=["Text"])
            )
            vision_description = response.text.strip()
            self.finished.emit(vision_description)
        except Exception as e:
            self.error.emit(str(e))

# -------------------------------------------------------------------
# GLOBAL CONFIGURATION & CLIENT SETUP
GEMINI_API_KEY = "AIzaSyCH3tL5tm9Q3itFCTcjpUiyLWTzlgeUz3c"
FAL_KEY = "b8709a63-2006-4c43-8629-e30599c9ad10:b8bff8ab5d51d349eb43c5ab62625382"
os.environ["FAL_KEY"] = FAL_KEY

client = genai.Client(api_key=GEMINI_API_KEY)
gemini_model = "gemini-2.0-flash-exp"

# -------------------------------------------------------------------
# SAMPLE PLACEHOLDER PROMPTS (3D Focused)
SAMPLE_PROMPTS = [
    "Generate a 3D model of a futuristic spaceship with sleek curves.",
    "Create a 3D digital sculpture of a mythical creature with intricate details.",
    "Render a 3D design of a modern building with glass and steel elements."
]

# -------------------------------------------------------------------
# HELPER FUNCTIONS
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def save_base64_image(base64_string, file_path):
    img_data = base64.b64decode(base64_string)
    with open(file_path, "wb") as f:
        f.write(img_data)
    return file_path

# -------------------------------------------------------------------
# LOADING ANIMATION WIDGET (Pulsating & Rotating Circle)
class LoadingAnimationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.angle = 0
        self.scale = 1.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(50)
        self.setFixedSize(50, 50)
    
    def update_animation(self):
        self.angle = (self.angle + 15) % 360
        self.scale = 1.0 + 0.25 * math.sin(time.time() * 10)
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy = self.width() / 2, self.height() / 2
        painter.translate(cx, cy)
        painter.rotate(self.angle * 0.5)
        painter.scale(self.scale, self.scale)
        pen = QPen(QColor("#4a4fef"), 3)
        painter.setPen(pen)
        rect = QRectF(-15, -15, 30, 30)
        painter.drawEllipse(rect)

# -------------------------------------------------------------------
# LOADING MESSAGE WIDGET
class LoadingMessageWidget(QWidget):
    def __init__(self, message="Processing...", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        self.spinner = LoadingAnimationWidget()
        layout.addWidget(self.spinner)
        label = QLabel(message)
        label.setStyleSheet("font-size: 14px;")
        layout.addWidget(label)
        self.setLayout(layout)

# -------------------------------------------------------------------
# STANDARD MESSAGE WIDGET (User and AI messages)
class ImageMessageWidget(QWidget):
    convert_to_3d_signal = pyqtSignal(str)
    
    def __init__(self, text: str, image_data: Union[str, List[str]], is_user: bool = False, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        self.image_data = image_data  # single image path or list of paths
        self.vision_description = ""  # to be updated later with vision info
        self.text_label = None
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        if is_user:
            margin_layout = QHBoxLayout()
            margin_layout.addStretch(1)
            content_widget = QWidget()
            content_layout = QVBoxLayout(content_widget)
            content_layout.setContentsMargins(0, 0, 0, 0)
            margin_layout.addWidget(content_widget, 4)
            layout.addLayout(margin_layout)
        else:
            content_widget = self
            content_layout = layout
        
        bubble = QFrame(content_widget)
        if is_user:
            bubble.setStyleSheet("background-color: #222222; border-radius: 16px; padding: 10px;")
        else:
            bubble.setStyleSheet("background: transparent;")
        bubble_layout = QVBoxLayout(bubble)
        
        if text:
            self.text_label = QLabel(text)
            self.text_label.setWordWrap(True)
            self.text_label.setStyleSheet("font-size: 14px;")
            bubble_layout.addWidget(self.text_label)
        
        if isinstance(self.image_data, list):
            grid_layout = QGridLayout()
            grid_layout.setSpacing(5)
            n = len(self.image_data)
            rows = ceil(sqrt(n))
            cols = ceil(n / rows)
            cell_size = int(300 / cols)
            for i, path in enumerate(self.image_data):
                label = ClickableLabel()
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(cell_size, cell_size, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
                    label.setPixmap(scaled_pixmap)
                label.setFixedSize(cell_size, cell_size)
                label.setStyleSheet("border: 1px solid #666666;")
                label.clicked.connect(lambda p=path: self.show_image_popup(p))
                grid_layout.addWidget(label, i // cols, i % cols)
            bubble_layout.addLayout(grid_layout)
        else:
            container = QWidget()
            container.setStyleSheet("background: transparent;")
            container.setFixedSize(400, 300)
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            self.image_label = ClickableLabel()
            pixmap = QPixmap(self.image_data)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(400, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.image_label.setPixmap(pixmap)
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image_label.setFixedSize(400, 300)
            self.image_label.clicked.connect(lambda: self.show_image_popup(self.image_data))
            container_layout.addWidget(self.image_label)
            self.download_btn = QPushButton("", container)
            self.download_btn.setIcon(get_download_icon())
            self.download_btn.setIconSize(QSize(24, 24))
            self.download_btn.setStyleSheet("""
                QPushButton {
                    background-color: rgba(0,0,0,0.5);
                    border: none;
                }
                QPushButton:hover {
                    background-color: rgba(0,0,0,0.7);
                }
            """)
            self.download_btn.setFixedSize(30, 30)
            self.download_btn.move(container.width() - self.download_btn.width() - 5, 5)
            self.download_btn.clicked.connect(lambda: self.download_image(self.image_data))
            self.download_btn.hide()
            container.installEventFilter(self)
            bubble_layout.addWidget(container)
        
        if is_user:
            content_layout.addWidget(bubble)
        else:
            content_layout.addWidget(bubble)
            content_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
    
    def eventFilter(self, source, event):
        if hasattr(self, 'download_btn'):
            if event.type() == event.Type.Enter:
                self.download_btn.show()
            elif event.type() == event.Type.Leave:
                self.download_btn.hide()
        return super().eventFilter(source, event)
    
    def show_image_popup(self, image_path):
        popup = ImagePopup(image_path, self.vision_description, self.window())
        center_point = self.window().rect().center()
        popup.move(center_point - popup.rect().center())
        popup.show()
    
    def download_image(self, image_path):
        default_dir = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DownloadLocation)
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", os.path.join(default_dir, os.path.basename(image_path)), "Image Files (*.png *.jpg *.jpeg *.webp)")
        if save_path:
            try:
                shutil.copy(image_path, save_path)
            except Exception as e:
                QMessageBox.critical(self, "Download Error", str(e))
    
    def update_vision_description(self, vision_desc: str):
        self.vision_description = vision_desc

# -------------------------------------------------------------------
# EMBEDDED 3D PREVIEW WIDGET (With Overlay Buttons on Hover)
class Model3DMessageWidget(QWidget):
    def __init__(self, model_url, parent=None):
        super().__init__(parent)
        self.model_url = model_url
        self.setMinimumHeight(320)
        self.setStyleSheet("background-color: transparent;")
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.web_view = QWebEngineView()
        self.web_view.setMinimumHeight(300)
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>3D Model Viewer</title>
            <style>
                body {{ margin: 0; padding: 0; overflow: hidden; background-color: #1a1a1a; }}
                canvas {{ width: 100%; height: 100%; display: block; }}
            </style>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.min.js"></script>
        </head>
        <body>
            <div id="viewer"></div>
            <script>
                let scene, camera, renderer, controls, model;
                function init() {{
                    scene = new THREE.Scene();
                    scene.background = new THREE.Color(0x1a1a1a);
                    camera = new THREE.PerspectiveCamera(75, window.innerWidth / (window.innerHeight-60), 0.1, 1000);
                    camera.position.z = 5;
                    renderer = new THREE.WebGLRenderer({{ antialias: true }});
                    renderer.setSize(window.innerWidth, window.innerHeight-60);
                    document.getElementById('viewer').appendChild(renderer.domElement);
                    controls = new THREE.OrbitControls(camera, renderer.domElement);
                    controls.enableDamping = true;
                    controls.dampingFactor = 0.05;
                    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
                    scene.add(ambientLight);
                    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
                    directionalLight.position.set(1, 1, 1);
                    scene.add(directionalLight);
                    const loader = new THREE.GLTFLoader();
                    loader.load(
                        '{self.model_url}',
                        function(gltf) {{
                            model = gltf.scene;
                            scene.add(model);
                            const box = new THREE.Box3().setFromObject(model);
                            const center = box.getCenter(new THREE.Vector3());
                            model.position.x = -center.x;
                            model.position.y = -center.y;
                            model.position.z = -center.z;
                            const size = box.getSize(new THREE.Vector3());
                            const maxDim = Math.max(size.x, size.y, size.z);
                            camera.position.z = maxDim * 2;
                            const gridHelper = new THREE.GridHelper(maxDim * 2, 20);
                            scene.add(gridHelper);
                        }},
                        function(xhr) {{
                            console.log((xhr.loaded / xhr.total * 100) + '% loaded');
                        }},
                        function(error) {{
                            console.error('Error loading model:', error);
                        }}
                    );
                    window.addEventListener('resize', onWindowResize, false);
                    animate();
                }}
                function onWindowResize() {{
                    camera.aspect = window.innerWidth / (window.innerHeight-60);
                    camera.updateProjectionMatrix();
                    renderer.setSize(window.innerWidth, window.innerHeight-60);
                }}
                function animate() {{
                    requestAnimationFrame(animate);
                    controls.update();
                    renderer.render(scene, camera);
                }}
                init();
            </script>
        </body>
        </html>
        """
        self.web_view.setHtml(html_content)
        self.main_layout.addWidget(self.web_view)
        
        self.overlay = QWidget(self)
        self.overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.overlay.setStyleSheet("background: transparent;")
        self.overlay.setGeometry(self.rect())
        self.download_btn = QPushButton("", self.overlay)
        self.download_btn.setIcon(get_download_icon())
        self.download_btn.setIconSize(QSize(24, 24))
        self.download_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(176,176,176,0.8);
                border: none;
            }
            QPushButton:hover {
                background-color: rgba(160,160,160,0.8);
            }
        """)
        self.download_btn.setFixedSize(30, 30)
        self.download_btn.move(self.width() - self.download_btn.width() - 5, 5)
        self.download_btn.clicked.connect(self.download_model)
        self.download_btn.hide()
        self.installEventFilter(self)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.overlay.setGeometry(self.rect())
        self.download_btn.move(self.width() - self.download_btn.width() - 5, 5)
    
    def eventFilter(self, source, event):
        if event.type() == event.Type.Enter:
            self.download_btn.show()
        elif event.type() == event.Type.Leave:
            self.download_btn.hide()
        return super().eventFilter(source, event)
    
    def download_model(self):
        default_dir = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DownloadLocation)
        save_path, _ = QFileDialog.getSaveFileName(self, "Save 3D Model", os.path.join(default_dir, os.path.basename(self.model_url)), "All Files (*)")
        if save_path:
            try:
                response = requests.get(self.model_url)
                if response.status_code == 200:
                    with open(save_path, "wb") as f:
                        f.write(response.content)
            except Exception as e:
                QMessageBox.critical(self, "Download Error", str(e))

# -------------------------------------------------------------------
# MAIN APPLICATION WINDOW
class OmniDirectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Omni Director")
        self.setMinimumSize(1200, 800)
        self.setFont(QFont("Poppins", 10))
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QHBoxLayout(self.central_widget)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self.chat_widget = QWidget()
        chat_layout = QVBoxLayout(self.chat_widget)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_content = QWidget()
        self.messages_layout = QVBoxLayout(self.scroll_content)
        self.messages_layout.addStretch()
        self.scroll_area.setWidget(self.scroll_content)
        chat_layout.addWidget(self.scroll_area)
        
        # Bottom input area with increased height (100px)
        input_container = QWidget()
        input_container.setFixedHeight(100)
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(2)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: #2d2d30;
                height: 4px;
            }
            QProgressBar::chunk {
                background-color: #4a4fef;
            }
        """)
        input_layout.addWidget(self.progress_bar)
        
        # Input frame with integrated text input and buttons.
        input_frame = QFrame()
        input_frame.setStyleSheet("background-color: #2d2d30; border-radius: 10px;")
        input_frame_layout = QVBoxLayout(input_frame)
        input_frame_layout.setContentsMargins(10, 6, 10, 6)
        input_frame_layout.setSpacing(5)
        
        self.input_text = QTextEdit()
        self.input_text.setFixedHeight(40)
        self.input_text.setPlaceholderText("Type your instructions here...")
        self.input_text.setStyleSheet("""
            QTextEdit {
                background-color: transparent;
                color: #f0f0f0;
                border: none;
                padding: 0;
                font-size: 14px;
            }
        """)
        input_frame_layout.addWidget(self.input_text)
        
        self.typewriter = TypewriterThread(SAMPLE_PROMPTS)
        self.typewriter.update_text.connect(self.update_placeholder)
        self.typewriter.start()
        
        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(8)
        
        self.upload_btn = QPushButton("Attach Images")
        self.upload_btn.setIcon(get_dark_grey_file_icon())
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #666666;
                border-radius: 8px;
                padding: 5px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
        """)
        self.upload_btn.clicked.connect(self.upload_image)
        buttons_layout.addWidget(self.upload_btn)
        
        self.image_preview_container = QWidget()
        self.image_preview_layout = QHBoxLayout(self.image_preview_container)
        self.image_preview_layout.setContentsMargins(0, 0, 0, 0)
        self.image_preview_layout.setSpacing(5)
        buttons_layout.addWidget(self.image_preview_container)
        
        buttons_layout.addStretch()
        
        self.send_btn = QPushButton("Send")
        self.send_btn.setIcon(QIcon.fromTheme("send"))
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #666666;
                border-radius: 8px;
                padding: 5px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
        """)
        self.send_btn.clicked.connect(self.send_message)
        buttons_layout.addWidget(self.send_btn)
        
        input_frame_layout.addLayout(buttons_layout)
        input_layout.addWidget(input_frame, 1)
        
        chat_layout.addWidget(input_container)
        chat_layout.setAlignment(input_container, Qt.AlignmentFlag.AlignBottom)
        chat_layout.setContentsMargins(10, 10, 10, 5)
        
        self.info_widget = QWidget()
        info_layout = QVBoxLayout(self.info_widget)
        self.status_label = QLabel("Omni Director - Ready")
        self.status_label.setStyleSheet("color: #4a4fef; font-weight: bold; font-size: 16px;")
        info_layout.addWidget(self.status_label)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("""
            QTextEdit {
                background-color: #1d1d20;
                color: #a0a0a0;
                border-radius: 10px;
                padding: 10px;
                font-family: monospace;
                font-size: 12px;
            }
        """)
        info_layout.addWidget(self.log_area)
        
        self.splitter.addWidget(self.chat_widget)
        self.splitter.addWidget(self.info_widget)
        self.splitter.setSizes([700, 300])
        main_layout.addWidget(self.splitter)
        self.apply_dark_theme()
        
        self.chat_history = []
        self.generated_images = []
        self.current_image_paths = []
        self.temp_dir = Path("./temp")
        self.temp_dir.mkdir(exist_ok=True)
        self.loading_widget = None
        self.current_3d_url = None
        self.vision_workers = []
        self.log("Omni Director initialized and ready.")
        self.log("Attach images and provide instructions to begin.")
    
    def apply_dark_theme(self):
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 33))
        dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(240, 240, 240))
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 28))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(35, 35, 38))
        dark_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(240, 240, 240))
        dark_palette.setColor(QPalette.ColorRole.ToolTipText, QColor(240, 240, 240))
        dark_palette.setColor(QPalette.ColorRole.Text, QColor(240, 240, 240))
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(45, 45, 48))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(240, 240, 240))
        dark_palette.setColor(QPalette.ColorRole.Link, QColor(74, 79, 239))
        dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(74, 79, 239))
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(240, 240, 240))
        QApplication.setPalette(dark_palette)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e21;
                color: #f0f0f0;
                font-family: 'Poppins', sans-serif;
            }
            QScrollArea, QScrollBar {
                background-color: #1e1e21;
                border: none;
            }
            QScrollBar:vertical {
                width: 12px;
                background: #1e1e21;
            }
            QScrollBar::handle:vertical {
                background: #3d3d40;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
    
    def update_placeholder(self, text):
        self.input_text.setPlaceholderText(text)
    
    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def upload_image(self):
        file_dialog = QFileDialog()
        initial_dir = getattr(self, "last_directory", "")
        file_paths, _ = file_dialog.getOpenFileNames(self, "Select Images", initial_dir, "Image Files (*.png *.jpg *.jpeg *.webp)")
        if file_paths:
            if len(file_paths) > 5:
                file_paths = file_paths[:5]
            self.current_image_paths = file_paths
            self.last_directory = os.path.dirname(file_paths[0])
            self.clear_layout(self.image_preview_layout)
            for path in file_paths:
                thumb = QLabel()
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    pixmap = pixmap.scaled(50, 50, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
                    thumb.setPixmap(pixmap)
                thumb.setFixedSize(50, 50)
                thumb.setStyleSheet("border: 1px solid #666666;")
                self.image_preview_layout.addWidget(thumb)
            file_names = [os.path.basename(fp) for fp in file_paths]
            self.log(f"Images attached: {', '.join(file_names)}")
    
    def send_message(self):
        message = self.input_text.toPlainText().strip()
        if not message:
            QMessageBox.warning(self, "Empty Message", "Please enter some instructions.")
            return
        attached_images = self.current_image_paths if self.current_image_paths else []
        self.chat_history.append({
            "role": "user",
            "text": message,
            "images": attached_images,
            "timestamp": time.time()
        })
        if attached_images:
            image_to_send = attached_images if len(attached_images) > 1 else attached_images[0]
        else:
            image_to_send = None
        self.add_message(message, image_to_send, is_user=True)
        self.input_text.clear()
        self.current_image_paths = []
        self.clear_layout(self.image_preview_layout)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.disable_input(True)
        self.log(f"Sending text refinement request. Images provided: {bool(attached_images)}")
        self.loading_widget = LoadingMessageWidget("Refining prompt...")
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, self.loading_widget)
        QTimer.singleShot(100, self.scroll_to_bottom)
        self.text_worker = TextRefinementWorker(self.chat_history, message)
        self.text_worker.finished.connect(self.handle_text_refinement)
        self.text_worker.error.connect(self.handle_error)
        self.text_worker.start()
    
    def handle_text_refinement(self, refined_prompt):
        self.log(f"Text refinement result: {refined_prompt}")
        if self.loading_widget:
            self.messages_layout.removeWidget(self.loading_widget)
            self.loading_widget.deleteLater()
            self.loading_widget = None
        self.log(f"Final refined prompt from text model: {refined_prompt}")
        lower_prompt = refined_prompt.lower()
        self.log(f"Refined prompt (lowercase): {lower_prompt}")
        multi_image = (("together" in lower_prompt) or ("togehter" in lower_prompt) or ("2 dogs" in lower_prompt)) and ("dog" in lower_prompt)
        self.log(f"Multi-image condition: {multi_image}")
        if multi_image:
            images_to_attach = []
            sorted_history = sorted(self.chat_history, key=lambda msg: msg.get("timestamp", 0))
            for msg in sorted_history:
                if msg.get("images"):
                    images_to_attach.extend(msg["images"])
            images_to_attach = list(dict.fromkeys(images_to_attach))
            self.log(f"Detected multi-image request; attaching {len(images_to_attach)} image(s): {images_to_attach}")
        else:
            images_to_attach = self.chat_history[-1].get("images", [])
            self.log(f"Attaching {len(images_to_attach)} image(s) based on current message: {images_to_attach}")
        self.loading_widget = LoadingMessageWidget("Generating image...")
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, self.loading_widget)
        QTimer.singleShot(100, self.scroll_to_bottom)
        self.image_worker = ImageGenerationWorker(refined_prompt, image_paths=images_to_attach)
        self.image_worker.progress.connect(self.update_progress)
        self.image_worker.finished.connect(self.handle_image_generation)
        self.image_worker.error.connect(self.handle_error)
        self.image_worker.start()
    
    def handle_image_generation(self, result):
        self.progress_bar.setVisible(False)
        self.disable_input(False)
        if self.loading_widget:
            self.messages_layout.removeWidget(self.loading_widget)
            self.loading_widget.deleteLater()
            self.loading_widget = None
        text = result.get("text", "Generated image")
        images = result.get("images", [])
        self.log(f"Image generation result received. Text: {text}")
        if not images:
            self.add_message("I couldn't generate an image. Please try again.", None, is_user=False)
            self.log("Error: No image was generated")
            return
        saved_images = []
        for i, img_data in enumerate(images):
            file_name = f"gemini_generated_{int(time.time())}_{i}.png"
            file_path = str(self.temp_dir / file_name)
            with open(file_path, "wb") as f:
                f.write(img_data)
            saved_images.append(file_path)
            self.log(f"Generated image saved as {file_name}")
        self.chat_history.append({
            "role": "ai",
            "text": text,
            "images": saved_images,
            "timestamp": time.time()
        })
        self.add_message(text, saved_images if len(saved_images) > 1 else saved_images[0], is_user=False)
        vision_worker = VisionWorker(saved_images)
        self.vision_workers.append(vision_worker)
        message_index = len(self.chat_history) - 1
        def on_finished(vision_desc, idx=message_index, worker=vision_worker):
            if worker in self.vision_workers:
                self.vision_workers.remove(worker)
        vision_worker.finished.connect(on_finished)
        vision_worker.error.connect(lambda err, worker=vision_worker: (self.handle_error(err),
                                                                       self.vision_workers.remove(worker) if worker in self.vision_workers else None))
        vision_worker.start()
    
    def add_message(self, text, image_data: Union[str, List[str]], is_user=False):
        if image_data == "3d_preview":
            widget = Model3DMessageWidget(self.current_3d_url)
        else:
            widget = ImageMessageWidget(text, image_data, is_user)
            if (not is_user) and image_data not in [None, "3d_preview"]:
                self.generated_images.append(image_data)
                if not is_user:
                    widget.convert_to_3d_signal.connect(self.start_3d_conversion)
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, widget)
        QTimer.singleShot(100, self.scroll_to_bottom)
    
    def scroll_to_bottom(self):
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        cursor = self.log_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        if "error" in message.lower():
            fmt.setForeground(QColor("#ff6b6b"))
        elif "complete" in message.lower() or "success" in message.lower():
            fmt.setForeground(QColor("#4ecca3"))
        else:
            fmt.setForeground(QColor("#a0a0a0"))
        cursor.insertText(log_entry + "\n", fmt)
        self.log_area.setTextCursor(cursor)
        self.log_area.ensureCursorVisible()
        print(log_entry)
    
    def disable_input(self, disable: bool):
        self.input_text.setDisabled(disable)
        self.send_btn.setDisabled(disable)
        self.upload_btn.setDisabled(disable)
    
    def handle_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.disable_input(False)
        self.log(f"Error: {error_message}")
        QMessageBox.critical(self, "Error", error_message)
        if self.loading_widget:
            self.messages_layout.removeWidget(self.loading_widget)
            self.loading_widget.deleteLater()
            self.loading_widget = None
    
    def start_3d_conversion(self, image_path):
        self.log(f"Convert-to-3D requested for image: {image_path}")
        try:
            public_url = fal_client.upload_file(image_path)
            self.log(f"Fal-client upload success. URL: {public_url}")
        except Exception as e:
            self.log(f"Error uploading image for 3D conversion: {str(e)}")
            return
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.disable_input(True)
        self.loading_widget = LoadingMessageWidget("Converting to 3D...")
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, self.loading_widget)
        QTimer.singleShot(100, self.scroll_to_bottom)
        self.fal_worker = FalWorker(public_url)
        self.fal_worker.progress.connect(self.update_progress)
        self.fal_worker.finished.connect(self.handle_fal_result)
        self.fal_worker.error.connect(self.handle_error)
        self.fal_worker.log_message.connect(self.log)
        self.fal_worker.start()
    
    def handle_fal_result(self, result):
        self.progress_bar.setVisible(False)
        self.disable_input(False)
        if self.loading_widget:
            self.messages_layout.removeWidget(self.loading_widget)
            self.loading_widget.deleteLater()
            self.loading_widget = None
        self.log("FalWorker result received.")
        model_url = result.get("model_url")
        if model_url:
            self.current_3d_url = model_url
            self.add_message("", "3d_preview", is_user=False)
        else:
            self.log("Error: No model URL returned from 3D conversion.")

# -------------------------------------------------------------------
# MAIN ENTRY POINT
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Poppins", 10))
    window = OmniDirectorApp()
    window.show()
    sys.exit(app.exec())
