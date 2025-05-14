const { contextBridge } = require('electron');
const path = require('path');
const { spawn, execFile } = require('child_process');

contextBridge.exposeInMainWorld('electron', {
    inputImages: async () => {
        const fileInput = document.getElementById('img-file');
        const imgPathsDisplay = document.getElementById('img-paths');

        // Show only the file names
        const fileNames = Array.from(fileInput.files).map(file => file.name).join('<br>');

        // Display the file names in the img-paths div
        imgPathsDisplay.innerHTML = fileNames || 'No files selected';
    },
    
    normalizeImages: async () => {
        const detectionStatus = document.getElementById('detection-status');
        const predictBtn = document.getElementById('predict-btn');
        const fileInput = document.getElementById('img-file');
        const loaderIcon = document.getElementById('loader');
        const loaderText = document.getElementById('loader-text');
        const selectedFiles = Array.from(fileInput.files).map(file => file.path);
        const predStatsContainer = document.getElementById('stats-chart');
        const totalCapsids = document.getElementById('num-total-capsids');
        const fullCapsids = document.getElementById('num-full-capsids');
        const partialCapsids = document.getElementById('num-partial-capsids');
        const emptyCapsids = document.getElementById('num-empty-capsids');
        const viableFraction = document.getElementById('viable-fraction');
        const aggregation = document.getElementById('aggregation-value');
        const ice = document.getElementById('ice-value');
        let patchingInputs = [];
        let segmentingInputs = [];
        let classifyingInputs = [];
        let updatingPredictionsInputs = [];

        detectionStatus.innerHTML = 'Detecting Capsids. See Below Results';
        predictBtn.disabled = true;
        predictBtn.textContent = 'Detecting...';

        // Show loader icon
        loaderIcon.style.display = 'block';
        loaderText.innerHTML = 'Normalizing Images...';
        loaderText.style.display = 'block';

        if (selectedFiles.length === 0) {
            console.log('No files were selected');
            return;
        }

        console.log("Selected files:", selectedFiles);

        // Define the path to the Python script
        const scriptPath = path.join(__dirname, '../../engine/normalizing.py');
        
        // Spawn a child process to run the Python script with the selected files as arguments
        const pythonProcess1 = spawn('python', [scriptPath, ...selectedFiles]);

        // // Define the path to the Python script
        // const scriptPath = path.join(__dirname, '../../engine/normalizing');
        // console.log("Script path:", scriptPath);

        // // Spawn a child process to run the Python script with the selected files as arguments
        // const pythonProcess1 = execFile(scriptPath, [...selectedFiles]);

        pythonProcess1.stdout.on('data', (data) => {
            const output = data.toString().trim();
            console.log(`normalizing.py output: ${output}`);
            const resultContainer = document.getElementById('result');

            // hide the loader icon
            // resultContainer.innerHTML = ''; // Clear previous images
            loaderIcon.style.display = 'none';
            loaderText.style.display = 'none';

            const normalizedPyOutput = output.split('\n');
            console.log("Processed Image Paths:", normalizedPyOutput);

            normalizedPyOutput.forEach(img_path1 => { 
                if (img_path1.startsWith('/') && img_path1.includes('.jpg')) {
                    // displaying the normalized images in the results section
                    // path to individual normalized images
                    // console.log("Normalized image path:",`file://${img_path1}`);
                    const imgElement = document.createElement('img');
                    imgElement.src = `file://${img_path1}`;
                    imgElement.alt = "Processed Image";

                    resultContainer.appendChild(imgElement);
                    segmentingInputs.push(img_path1);
                    updatingPredictionsInputs.push(img_path1);

                } else if ((img_path1.startsWith('/') && !img_path1.includes('.jpg'))) {
                    // path to normalized images (just the folder path)
                    // save this path to patches_inputs
                    patchingInputs.push(img_path1); 
                } else if (img_path1.includes('sum_image_areas:')) {
                    const imageAreaValue = img_path1.replace('sum_image_areas:', '');
                    updatingPredictionsInputs.unshift(imageAreaValue);
                } else {
                    // console.log(img_path1); // this is not usually a path, but instead a message from the Python script
                }
            });

            // show the loader icon
            loaderIcon.style.display = 'block';
            loaderText.innerHTML = 'Segmenting Images...';
            loaderText.style.display = 'block';

            console.log("After Process 1 Segmenting inputs:", segmentingInputs);
            console.log("After Process 1 Patching inputs:", patchingInputs);

            // Spawn a child process to run the Python script with the data from normalizing.py as arguments into segmenting.py
            const scriptPath2 = path.join(__dirname, '../../engine/segmenting.py');
            const pythonProcess2 = spawn('python', [scriptPath2, ...segmentingInputs]);
            // const scriptPath2 = path.join(__dirname, '../../engine/segmenting_bundle/segmenting');
            // const pythonProcess2 = execFile(scriptPath2, [...segmentingInputs]);

            pythonProcess2.stdout.on('data', (data2) => {
                const output2 = data2.toString().trim();
                console.log(`segmenting.py stdout: ${output2}`);

                // hide the loader icon
                loaderIcon.style.display = 'none';
                loaderText.style.display = 'none';

                const segmentingPyOutput = output2.split('\n');

                segmentingPyOutput.forEach(img_path2 => {
                    if (img_path2.startsWith('/')) {
                        if (img_path2.includes('annotated')) {
                            const imgElement = document.createElement('img');
                            imgElement.src = `file://${img_path2}`;
                            console.log("Segmented image path:",`file://${img_path2}`);
                            imgElement.alt = "Segmented Image";

                            resultContainer.appendChild(imgElement);
                        } else if (img_path2.includes('sam_results_by_image')) {
                            console.log("Sam results by image path:",`file://${img_path2}`);
                            updatingPredictionsInputs.unshift(img_path2);
                        } else {
                            // path to coco annotations
                            console.log("COCO sam results path:",`file://${img_path2}`);
                            patchingInputs.unshift(img_path2);
                            updatingPredictionsInputs.unshift(img_path2);
                        }
                    } else {
                        if (img_path2.includes('script message:')) {
                            // console.log(img_path2);
                        } else {
                            // console.log("Total Capsids:", img_path2);
                            // totalCapsids.innerHTML = img_path2;
                        }
                    }
                });

                console.log("After Process 2 Patching inputs:", patchingInputs);
                console.log("After Process 2 updating inputs:", updatingPredictionsInputs);

                // show the loader icon
                loaderIcon.style.display = 'block';
                loaderText.innerHTML = 'Patching Images...';
                loaderText.style.display = 'block';

                // Spawn a child process to run the Python script patching.py
                const scriptPath3 = path.join(__dirname, '../../engine/patching.py');
                const pythonProcess3 = spawn('python', [scriptPath3, ...patchingInputs]);
                // const scriptPath3 = path.join(__dirname, '../../engine/patching');
                // const pythonProcess3 = execFile(scriptPath3, [...patchingInputs]);

                pythonProcess3.stdout.on('data', (data3) => {
                    const output3 = data3.toString().trim();
                    console.log(`patching.py stdout: ${output3}`);

                    // hide the loader icon
                    loaderIcon.style.display = 'none';
                    loaderText.style.display = 'none';

                    const patchingPyOutput = output3.split('\n');

                    patchingPyOutput.forEach(img_path3 => {
                        if (img_path3.startsWith('/')) {
                            classifyingInputs.push(img_path3);
                        } else {
                            // console.log(img_path3);
                        }
                    });

                    console.log("After Process 3 Classifying inputs:", classifyingInputs);

                    // show the loader icon
                    loaderIcon.style.display = 'block';
                    loaderText.innerHTML = 'Classifying Images...';
                    loaderText.style.display = 'block';

                    // Spawn a child process to run the Python script classifying.py
                    const scriptPath4 = path.join(__dirname, '../../engine/classifyingIA.py');
                    const pythonProcess4 = spawn('python', [scriptPath4, ...classifyingInputs]);
                    // const scriptPath4 = path.join(__dirname, '../../engine/classifyingIA');
                    // const pythonProcess4 = execFile(scriptPath4, [...classifyingInputs]);

                    pythonProcess4.stdout.on('data', (data4) => {
                        const output4 = data4.toString().trim();
                        console.log(`classifyingIA.py stdout: ${output4}`);

                        // hide the loader icon
                        loaderIcon.style.display = 'none';
                        loaderText.style.display = 'none';

                        const classifyingPyOutput = output4.split('\n');

                        classifyingPyOutput.forEach(img_path4 => {
                            if (img_path4.startsWith('/') && !img_path4.includes('.png')) {
                                // predicting data
                                updatingPredictionsInputs.unshift(img_path4);
                            } else if (img_path4.startsWith('/') && img_path4.includes('.png')) {
                                const imgElement = document.createElement('img');
                                imgElement.src = `file://${img_path4}`;
                                console.log("capsid count image path:",`file://${img_path4}`);
                                imgElement.alt = "Bar Graph of Capsid Counts Image";
                                predStatsContainer.appendChild(imgElement);
                            } else if (img_path4.includes('full_capsids:')) {
                                const fullCapsidsValue = img_path4.replace('full_capsids:', '');
                                fullCapsids.innerText = fullCapsidsValue;
                            } else if (img_path4.includes('partial_capsids:')) {
                                const partialCapsidsValue = img_path4.replace('partial_capsids:', '');
                                partialCapsids.innerText = partialCapsidsValue;
                            } else if (img_path4.includes('empty_capsids:')) {
                                const emptyCapsidsValue = img_path4.replace('empty_capsids:', '');
                                emptyCapsids.innerText = emptyCapsidsValue;
                            } else if (img_path4.includes('total_capsids:')) {
                                // remove the total_capsids: prefix and then update the total capsids count
                                const totalCapsidsValue = img_path4.replace('total_capsids:', '');
                                totalCapsids.innerText = totalCapsidsValue;
                            } else if (img_path4.includes('viable_fraction:')) {
                                // remove the viable_fraction: prefix and then update the viable fraction
                                const viableFractionValue = img_path4.replace('viable_fraction:', '');
                                viableFraction.innerText = viableFractionValue + ' %';
                            } else {
                                console.log(img_path4);
                            }
                        });

                        console.log("After Process 4 Updating predictions inputs:", updatingPredictionsInputs);

                        // show the loader icon
                        loaderIcon.style.display = 'block';
                        loaderText.innerHTML = 'Updating Predictions...';
                        loaderText.style.display = 'block';

                        // Spawn a child process to run the Python script updating.py
                        const scriptPath5 = path.join(__dirname, '../../engine/updating.py');
                        const pythonProcess5 = spawn('python', [scriptPath5, ...updatingPredictionsInputs]);
                        // const scriptPath5 = path.join(__dirname, '../../engine/updating');
                        // const pythonProcess5 = execFile(scriptPath5, [...updatingPredictionsInputs]);

                        pythonProcess5.stdout.on('data', (data5) => {
                            const output5 = data5.toString().trim();
                            console.log(`updating.py stdout: ${output5}`);

                            // hide the loader icon
                            loaderIcon.style.display = 'none';
                            loaderText.style.display = 'none';

                            const updatingPyOutput = output5.split('\n');

                            updatingPyOutput.forEach(img_path5 => {
                                if (img_path5.startsWith('/') && img_path5.includes('.json')) {
                                    console.log(img_path5);
                                    const linkDownload = document.createElement('a');
                                    linkDownload.href = `file://${img_path5}`;
                                    linkDownload.download = 'coco_annotations.json';

                                    const downloadButton = document.createElement('button');
                                    downloadButton.innerHTML = 'Download COCO Style JSON Annotations';
                                    // add file-download-btn id to the button
                                    downloadButton.id = 'file-download-btn';
                                    downloadButton.classList.add('btn');
                                    downloadButton.onclick = () => {
                                        linkDownload.click();
                                    };

                                    linkDownload.appendChild(downloadButton);
                                    resultContainer.appendChild(linkDownload);

                                } else if (img_path5.startsWith('/') && !img_path5.includes('.json')) {
                                    const imgElement = document.createElement('img');
                                    imgElement.src = `file://${img_path5}`;
                                    console.log("Segmented image path:",`file://${img_path5}`);
                                    imgElement.alt = "Segmented Image";

                                    resultContainer.appendChild(imgElement);
                                } else if (img_path5.includes('aggregation_area:')) {
                                    const aggregationValue = img_path5.replace('aggregation_area:', '');
                                    aggregation.innerText = aggregationValue;
                                } else if (img_path5.includes('ice_area:')) {
                                    const iceValue = img_path5.replace('ice_area:', '');
                                    ice.innerText = iceValue;
                                } else {
                                    console.log(img_path5);
                                }
                            });

                            // Show the predict button again
                            predictBtn.disabled = false;
                            // Change the button text back to "Detect"
                            predictBtn.textContent = 'Detect';
                            // Show the detection status
                            detectionStatus.innerHTML = 'Detection Completed. See Below Results';
                        });

                        pythonProcess5.stderr.on('data', (data5) => {
                            console.error(`stderr (updating): ${data5}`);
                        });

                        pythonProcess5.on('close', (code) => {
                            if (code !== 0) {
                                console.log(`Updating process exited with code ${code}`);
                                detectionStatus.innerHTML = 'Error while updating capsid labels. Please try again.';
                                // Hide the loader icon
                                loaderIcon.style.display = 'none';
                                loaderText.style.display = 'none';
                                // Show the predict button again
                                predictBtn.disabled = false;
                                // Change the button text back to "Detect"
                                predictBtn.textContent = 'Detect';
                            }
                        });

                    });

                    pythonProcess4.stderr.on('data', (data4) => {
                        console.error(`stderr (classifying): ${data4}`);
                    });

                    pythonProcess4.on('close', (code) => {
                        if (code !== 0) {
                            console.log(`Classifying process exited with code ${code}`);
                            detectionStatus.innerHTML = 'Error while classifying capsids. Please try again.';
                            // Hide the loader icon
                            loaderIcon.style.display = 'none';
                            loaderText.style.display = 'none';
                            // Show the predict button again
                            predictBtn.disabled = false;
                            // Change the button text back to "Detect"
                            predictBtn.textContent = 'Detect';
                        }
                    });
                });

                pythonProcess3.stderr.on('data', (data3) => {
                    console.error(`stderr (patching): ${data3}`);
                });

                pythonProcess3.on('close', (code) => {
                    if (code !== 0) {
                        console.log(`Patching process exited with code ${code}`);
                        detectionStatus.innerHTML = 'Error while creating capsid patches. Please try again.';
                        // Hide the loader icon
                        loaderIcon.style.display = 'none';
                        loaderText.style.display = 'none';
                        // Show the predict button again
                        predictBtn.disabled = false;
                        // Change the button text back to "Detect"
                        predictBtn.textContent = 'Detect';
                    }
                });

            });

            pythonProcess2.stderr.on('data', (data2) => {
                console.error(`stderr (segmenting): ${data2}`);
            });

            pythonProcess2.on('close', (code) => {
                if (code !== 0) {
                    console.log(`Segmenting process exited with code ${code}`);
                    detectionStatus.innerHTML = 'Error while segmenting capsids. Please try again.';
                    // Hide the loader icon
                    loaderIcon.style.display = 'none';
                    loaderText.style.display = 'none';
                    // Show the predict button again
                    predictBtn.disabled = false;
                    // Change the button text back to "Detect"
                    predictBtn.textContent = 'Detect';
                }
            });
        });

        pythonProcess1.stderr.on('data', (data) => {
            console.error(`stderr (normalizing): ${data}`);
        });

        pythonProcess1.on('close', (code) => {
            if (code !== 0) {
                console.log(`Normalizing exited with code ${code}`);
                detectionStatus.innerHTML = 'Error while normaling original images. Please try again. JPG images preferred';
                // Hide the loader icon
                loaderIcon.style.display = 'none';
                loaderText.style.display = 'none';
                // Show the predict button again
                predictBtn.disabled = false;
                // Change the button text back to "Detect"
                predictBtn.textContent = 'Detect';
            }
        });
    }
});
