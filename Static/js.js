// Updated js.js logic
console.log("JavaScript loaded successfully!");

function submitUserData() {
    const username = document.getElementById("username").value;
    const code = document.getElementById("code").value;
  
    fetch("http://127.0.0.1:8000/process-user/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ username, code }),
    })
      .then((response) => response.json())
      .then((data) => {
        const userResults = document.getElementById("user-results");
        const userStuff = document.getElementById("user-Stuff");
        userResults.innerText = data.message;
        userStuff.style.display = "block";
  
        const signed_in_user = document.getElementById("signed_in_user");
        const header1 = document.getElementById("header1");
        const hiddenContent = document.getElementById("hiddenContent");
        
        if (data.sign_in) { 
          signed_in_user.style.display = "block";
          header1.style.display = "block";
          hiddenContent.style.display = "block";
        } else {
          signed_in_user.style.display = "none";
          header1.style.display = "none";
          hiddenContent.style.display = "none";
        }

      })
      .catch((error) => console.error("Error:", error));
  }

const uploadButton = document.getElementById('uploadButton');
const fileInput = document.getElementById('fileInput');
const predictionResult = document.getElementById('predictionResult');
const fileDisplay = document.getElementById("fileDisplay"); 
const predictionList = document.getElementById('predictionList');
const prediction_results = document.getElementById('prediction_results');
const prediction_display = document.getElementById('prediction_display')

// Ensure required DOM elements exist
if (!uploadButton || !fileInput || !predictionResult || !prediction_display || !predictionList || !prediction_results) {
  console.error('One or more required elements are missing from the DOM.');
} else {
  uploadButton.addEventListener('click', async () => {
    var files = document.getElementById('fileInput').files;
    if (files.length > 0) {
        var fileList = '';
        var allResults = ''; // Initialize a string to collect all results
        var combinedFoundResults = []; // Initialize an array to store all found results

        for (var i = 0; i < files.length; i++) {
            const file = files[i];
            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }

                const data = await response.json();

                // Process the results for each file (image)
                allResults += `<h4>Results for ${file.name}:</h4>`;

                if (data.error) {
                    allResults += `<p>${data.error}</p>`;
                } else {
                    allResults += '<h5>Predictions:</h5>';
                    
                    // Display predictions for the current file
                    if (Array.isArray(data.predictions)) {
                        data.predictions.forEach(prediction => {
                            allResults += `<p>Prediction: ${prediction}</p>`;
                        });
                    } else {
                        allResults += '<p>No predictions available.</p>';
                    }

                    // Display found colors for the current file
                    if (data.foundColours) {
                        allResults += `<p>Found Colours: ${data.foundColours}</p>`;
                    } else {
                        allResults += '<p>No colours detected.</p>';
                    }

                    // Combine the found results from the current file with the overall array
                    if (Array.isArray(data.found_results)) {
                        combinedFoundResults = combinedFoundResults.concat(data.found_results);
                    }

                    // Display results for the current file
                    if (data.found_results) {
                        allResults += `<p>Found Results: ${data.found_results}</p>`;
                    } else {
                        allResults += '<p>No results available for the next step.</p>';
                    }
                }

                // Append all results at once after processing each file
                predictionResult.innerHTML = allResults;

            } catch (error) {
                console.error('Error:', error);
                predictionResult.innerHTML = `<p>An error occurred during upload.</p>`;
            }
        }

        // After processing all files, display the combined results
        if (combinedFoundResults.length > 0) {
            allResults += `<h5>Combined Found Results:</h5>`;
            allResults += `<p>${combinedFoundResults.join(', ')}</p>`; // Join all results into a single string
            predictionResult.innerHTML = allResults; // Append the combined results
        } else {
            allResults += `<h5>No combined results available.</h5>`;
            predictionResult.innerHTML = allResults;
        }
    } else {
        alert('Please select files to upload.');
    }
});
}

function getGreeting() {
  const name = document.getElementById('name').value;  // Get the name input
  const encodedName = encodeURIComponent(name);  // Encode the input to handle spaces and special characters
  const url = `http://localhost:8000/greet?name=${encodedName}`;  // URL to FastAPI endpoint

  // Use fetch to call the backend API
  fetch(url)
      .then(response => response.json())  // Parse the JSON response
      .then(data => {
          // Convert message to have line breaks by replacing newline characters with <br>
          const formattedMessage = data.message.replace(/\n/g, '<br>');  

          // Update the span with the formatted result
          document.getElementById('solve').innerHTML = formattedMessage;
      })
      .catch(error => {
          // Handle any errors
          console.error('Error:', error);
          document.getElementById('solve').textContent = 'Error occurred while fetching the data.';
      });
}