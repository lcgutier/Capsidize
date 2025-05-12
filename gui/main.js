// This app was built with the help of the following tutorials 
// Sourav Johar, "Making modern GUIs with Python and ElectronJS" https://youtu.be/627VBkAhKTc?si=JzKDmf03NtFYTM8Y
// Codevolution, "Electron js Tutorial - 2 - Hello World" https://youtu.be/tqBi_Tou6wQ?si=CnjgOsMkaZUBAUkc


const electron = require("electron"); // this is from the node modules package that we installed locally
const app = electron.app; // this is the application module of electron
const BrowserWindow = electron.BrowserWindow; // this is a class that creates a window
const ipcMain = electron.ipcMain; 
const dialog = electron.dialog;
const path = require("path"); // this will help us build the path to the file
const url = require("url"); // this is a built in url module that will help us navigate to the correct url

// here we will add the code that will create the window of the user interface
let win; // this is the reference to the window

// here we will define a function that is going to create a window
function createWindow() {
    win = new BrowserWindow({ width: 900, height: 700,
        webPreferences: {
            nodeIntegration: true, // this will allow us to use node modules in the front end
            contextIsolation: true, // this will allow us to use require in the front end
            preload: path.join(__dirname, 'linkers/preload.js')
    }});
    win.loadURL(url.format({
        pathname: path.join(__dirname, 'base.html'),  
        protocol: 'file', // we are serving the file not http
        slashes: true
    }))

    // win.webContents.openDevTools(); # You can uncomment this line to open the dev tools
    // now we will handle the event where the user closes the browser window
    win.on('closed', () => {
        win = null;
    });
};

app.on('ready', createWindow); // call only when ready or it might not act as expected

// because we are using mac we need to add this code 
app.on('window-all-closed', () => {
    app.quit()
});

app.on('activate', () => {
    if (win === null) {
        createWindow()
    }
});

ipcMain.handle('dialog:openFile', async () => {
    const result = await dialog.showOpenDialog(win, {
        properties: ['openFile', 'multiSelections'],
        filters: [{ name: 'Images', extensions: ['tif', 'tiff', 'jpg', 'jpeg', 'png'] }]
    });
    return result.filePaths;
});