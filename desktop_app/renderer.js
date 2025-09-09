const { ipcRenderer } = require('electron');

// API base URL - Real AI server
const API_URL = 'http://localhost:8900';

// State management
let currentFile = null;
let currentMode = 'photo';
let isProcessing = false;

// DOM elements
const dropZone = document.getElementById('drop-zone');
const previewContainer = document.getElementById('preview-container');
const processingOverlay = document.getElementById('processing-overlay');
const processingStatus = document.getElementById('processing-status');
const statusMessage = document.getElementById('status-message');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    initializeControls();
    initializeDropZone();
    checkBackendStatus();
});

// Tab switching
function initializeTabs() {
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetTab = tab.dataset.tab;
            
            // Update active tab
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            // Show corresponding controls
            document.querySelectorAll('.tab-content').forEach(content => {
                content.style.display = 'none';
            });
            document.getElementById(`${targetTab}-controls`).style.display = 'block';
            
            currentMode = targetTab;
        });
    });
}

// Initialize all controls
function initializeControls() {
    // Sliders
    const sliders = document.querySelectorAll('.slider');
    sliders.forEach(slider => {
        slider.addEventListener('input', (e) => {
            const valueSpan = document.getElementById(`${slider.id}-value`);
            if (valueSpan) {
                valueSpan.textContent = e.target.value;
            }
        });
    });
    
    // Preset buttons
    const presetBtns = document.querySelectorAll('.preset-btn');
    presetBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            presetBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            applyPreset(btn.dataset.preset);
        });
    });
    
    // Action buttons
    document.getElementById('browse-files').addEventListener('click', browseFiles);
    document.getElementById('process-photo').addEventListener('click', processPhoto);
    document.getElementById('reset-photo').addEventListener('click', resetPhoto);
    document.getElementById('save-photo').addEventListener('click', saveResult);
    document.getElementById('process-video').addEventListener('click', processVideo);
    document.getElementById('select-folder').addEventListener('click', selectFolder);
    document.getElementById('process-batch').addEventListener('click', processBatch);
    document.getElementById('auto-montage').addEventListener('click', autoMontage);
    document.getElementById('scene-detect').addEventListener('click', detectScenes);
}

// Drop zone functionality
function initializeDropZone() {
    dropZone.addEventListener('click', browseFiles);
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        
        const files = Array.from(e.dataTransfer.files);
        if (files.length > 0) {
            handleFileSelect(files[0].path);
        }
    });
}

// Browse files
async function browseFiles() {
    const filePath = currentMode === 'video' 
        ? await ipcRenderer.invoke('select-video')
        : await ipcRenderer.invoke('select-image');
    
    if (filePath) {
        handleFileSelect(filePath);
    }
}

// Handle file selection
function handleFileSelect(filePath) {
    currentFile = filePath;
    displayFile(filePath);
    statusMessage.textContent = `Loaded: ${filePath.split(/[\\/]/).pop()}`;
}

// Display file preview
function displayFile(filePath) {
    const isVideo = /\.(mp4|avi|mov|mkv|wmv|flv|webm)$/i.test(filePath);
    
    if (isVideo) {
        previewContainer.innerHTML = `
            <video class="preview-video" controls>
                <source src="file://${filePath}" type="video/mp4">
            </video>
        `;
    } else {
        previewContainer.innerHTML = `
            <img class="preview-image" src="file://${filePath}" alt="Preview" style="max-width: 100%; height: auto; object-fit: contain;">
        `;
    }
}

// Apply preset settings
function applyPreset(preset) {
    const presets = {
        cinematic: { strength: 85, skin: 70, grading: 80 },
        wedding: { strength: 60, skin: 95, grading: 40 },
        portrait: { strength: 70, skin: 98, grading: 30 },
        landscape: { strength: 90, skin: 0, grading: 70 },
        film: { strength: 75, skin: 60, grading: 90 },
        moody: { strength: 95, skin: 40, grading: 85 }
    };
    
    const settings = presets[preset];
    if (settings) {
        document.getElementById('ai-strength').value = settings.strength;
        document.getElementById('ai-strength-value').textContent = settings.strength;
        document.getElementById('skin-protection').value = settings.skin;
        document.getElementById('skin-protection-value').textContent = settings.skin;
        document.getElementById('color-grading').value = settings.grading;
        document.getElementById('color-grading-value').textContent = settings.grading;
    }
}

// Process photo with AI
async function processPhoto() {
    if (!currentFile || isProcessing) return;
    
    isProcessing = true;
    processingOverlay.classList.add('active');
    processingStatus.textContent = 'Loading AI models...';
    
    try {
        const formData = new FormData();
        const fileBlob = await fetch(`file://${currentFile}`).then(r => r.blob());
        formData.append('file', fileBlob, currentFile.split(/[\\/]/).pop());
        formData.append('ai_strength', document.getElementById('ai-strength').value);
        formData.append('skin_protection', document.getElementById('skin-protection').value);
        formData.append('color_grading', document.getElementById('color-grading').value);
        
        processingStatus.textContent = 'Processing with AI...';
        
        const response = await fetch(`${API_URL}/process/photo`, {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const result = await response.json();
            
            // Save original image URL for comparison
            const originalUrl = previewContainer.querySelector('img')?.src || currentFile;
            
            // Decode base64 image from response
            const imgData = 'data:image/jpeg;base64,' + result.image;
            
            // Create before/after view
            previewContainer.innerHTML = `
                <div style="display: flex; gap: 10px; width: 100%; height: 100%;">
                    <div style="flex: 1; text-align: center;">
                        <h3 style="color: #888; margin-bottom: 10px;">BEFORE</h3>
                        <img class="preview-image" src="${originalUrl}" alt="Original" style="max-width: 100%; height: auto; object-fit: contain; border: 2px solid #444;">
                    </div>
                    <div style="flex: 1; text-align: center;">
                        <h3 style="color: #4CAF50; margin-bottom: 10px;">AFTER (AI Enhanced)</h3>
                        <img class="preview-image" src="${imgData}" alt="Processed" style="max-width: 100%; height: auto; object-fit: contain; border: 2px solid #4CAF50;">
                    </div>
                </div>
            `;
            
            // Store processed image for saving
            window.processedImageData = imgData;
            
            statusMessage.textContent = `AI processing complete! ${result.message}`;
        } else {
            throw new Error('Processing failed');
        }
    } catch (error) {
        console.error('Processing error:', error);
        statusMessage.textContent = 'Error: ' + error.message;
    } finally {
        isProcessing = false;
        processingOverlay.classList.remove('active');
    }
}

// Process video with AI
async function processVideo() {
    if (!currentFile || isProcessing) return;
    
    isProcessing = true;
    processingOverlay.classList.add('active');
    processingStatus.textContent = 'Initializing video AI...';
    
    try {
        const formData = new FormData();
        formData.append('video_path', currentFile);
        formData.append('beat_sync', document.getElementById('beat-sync').value);
        
        processingStatus.textContent = 'Processing video with AI...';
        
        const response = await fetch(`${API_URL}/process/video`, {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const result = await response.json();
            statusMessage.textContent = `Video processed: ${result.output_path}`;
            displayFile(result.output_path);
        } else {
            throw new Error('Video processing failed');
        }
    } catch (error) {
        console.error('Video processing error:', error);
        statusMessage.textContent = 'Error: ' + error.message;
    } finally {
        isProcessing = false;
        processingOverlay.classList.remove('active');
    }
}

// Auto montage for videos
async function autoMontage() {
    if (!currentFile || isProcessing) return;
    
    isProcessing = true;
    processingOverlay.classList.add('active');
    processingStatus.textContent = 'Analyzing video scenes...';
    
    try {
        processingStatus.textContent = 'Creating AI montage...';
        
        const response = await fetch(`${API_URL}/video/montage`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_path: currentFile })
        });
        
        if (response.ok) {
            const result = await response.json();
            statusMessage.textContent = 'Auto montage complete!';
            displayFile(result.output_path);
        }
    } catch (error) {
        console.error('Montage error:', error);
        statusMessage.textContent = 'Error: ' + error.message;
    } finally {
        isProcessing = false;
        processingOverlay.classList.remove('active');
    }
}

// Detect scenes in video
async function detectScenes() {
    if (!currentFile || isProcessing) return;
    
    isProcessing = true;
    processingOverlay.classList.add('active');
    processingStatus.textContent = 'Detecting scenes with AI...';
    
    try {
        const response = await fetch(`${API_URL}/video/detect-scenes`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_path: currentFile })
        });
        
        if (response.ok) {
            const result = await response.json();
            statusMessage.textContent = `Found ${result.scenes.length} scenes`;
            console.log('Scenes:', result.scenes);
        }
    } catch (error) {
        console.error('Scene detection error:', error);
        statusMessage.textContent = 'Error: ' + error.message;
    } finally {
        isProcessing = false;
        processingOverlay.classList.remove('active');
    }
}

// Select folder for batch processing
async function selectFolder() {
    const folderPath = await ipcRenderer.invoke('select-folder');
    if (folderPath) {
        statusMessage.textContent = `Selected folder: ${folderPath}`;
        // Count files in folder
        try {
            const response = await fetch(`${API_URL}/batch/count`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ folder_path: folderPath })
            });
            
            if (response.ok) {
                const result = await response.json();
                document.getElementById('file-count').textContent = result.count;
            }
        } catch (error) {
            console.error('Error counting files:', error);
        }
    }
}

// Process batch
async function processBatch() {
    // Implementation for batch processing
    statusMessage.textContent = 'Batch processing started...';
}

// Reset photo
function resetPhoto() {
    if (currentFile) {
        displayFile(currentFile);
        statusMessage.textContent = 'Reset to original';
    }
}

// Save result
async function saveResult() {
    if (!window.processedImageData) {
        statusMessage.textContent = 'No processed image to save!';
        return;
    }
    
    const savePath = await ipcRenderer.invoke('save-file', 'processed_image.jpg');
    if (savePath) {
        try {
            // Convert base64 to buffer
            const base64Data = window.processedImageData.replace(/^data:image\/\w+;base64,/, '');
            const buffer = Buffer.from(base64Data, 'base64');
            
            // Write file using Node.js fs
            const fs = require('fs');
            fs.writeFileSync(savePath, buffer);
            
            statusMessage.textContent = `Saved to: ${savePath}`;
        } catch (error) {
            statusMessage.textContent = `Save error: ${error.message}`;
        }
    }
}

// Check backend status
async function checkBackendStatus() {
    const backendDot = document.getElementById('backend-status');
    const gpuDot = document.getElementById('gpu-status');
    const memoryDot = document.getElementById('memory-status');
    
    try {
        const response = await fetch(`${API_URL}/status`);
        if (response.ok) {
            const status = await response.json();
            backendDot.classList.remove('error');
            
            if (status.gpu_available) {
                gpuDot.classList.remove('error');
            } else {
                gpuDot.classList.add('error');
            }
            
            if (status.memory_usage < 80) {
                memoryDot.classList.remove('error');
            } else {
                memoryDot.classList.add('error');
            }
        } else {
            backendDot.classList.add('error');
        }
    } catch (error) {
        backendDot.classList.add('error');
        console.error('Backend connection error:', error);
    }
    
    // Check again in 5 seconds
    setTimeout(checkBackendStatus, 5000);
}