const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const express = require('express');

let mainWindow;
let pythonProcess;
let apiServer;

// Python backend server URL
const PYTHON_API_URL = 'http://localhost:8900';

// Start Python FastAPI backend (commented out - backend runs separately)
function startPythonBackend() {
  // Backend should be started separately before running the desktop app
  console.log('Connecting to Python backend at http://localhost:8900');
  
  // Check if backend is running
  const http = require('http');
  http.get('http://localhost:8900/', (res) => {
    console.log('Backend is running!');
  }).on('error', (err) => {
    console.error('Backend not found! Please start ai_photo_processor_real.py first');
  });
}

// Create the main application window
function createWindow() {
  // Disable GPU acceleration to fix GPU errors
  app.commandLine.appendSwitch('disable-gpu');
  app.commandLine.appendSwitch('disable-software-rasterizer');
  
  mainWindow = new BrowserWindow({
    width: 1600,
    height: 900,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      webSecurity: false,
      hardwareAcceleration: false
    },
    icon: path.join(__dirname, 'assets', 'icon.png'),
    titleBarStyle: 'hiddenInset',
    backgroundColor: '#1a1a1a'
  });

  mainWindow.loadFile('index.html');
  mainWindow.setMenuBarVisibility(false);
  
  // Open DevTools in development
  if (process.env.NODE_ENV === 'development') {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// App event handlers
app.whenReady().then(() => {
  startPythonBackend();
  
  // Wait a bit for Python server to start
  setTimeout(() => {
    createWindow();
  }, 3000);
});

app.on('window-all-closed', () => {
  // Backend runs separately, don't kill it
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});

// IPC handlers for file operations
ipcMain.handle('select-image', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'raw', 'dng'] }
    ]
  });
  
  if (!result.canceled) {
    return result.filePaths[0];
  }
  return null;
});

ipcMain.handle('select-video', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'Videos', extensions: ['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'] }
    ]
  });
  
  if (!result.canceled) {
    return result.filePaths[0];
  }
  return null;
});

ipcMain.handle('select-folder', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory']
  });
  
  if (!result.canceled) {
    return result.filePaths[0];
  }
  return null;
});

ipcMain.handle('save-file', async (event, defaultPath) => {
  const result = await dialog.showSaveDialog(mainWindow, {
    defaultPath: defaultPath,
    filters: [
      { name: 'JPEG', extensions: ['jpg', 'jpeg'] },
      { name: 'PNG', extensions: ['png'] },
      { name: 'Video', extensions: ['mp4', 'avi', 'mov'] }
    ]
  });
  
  if (!result.canceled) {
    return result.filePath;
  }
  return null;
});