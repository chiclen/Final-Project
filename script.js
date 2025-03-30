// Initialize webcam
const video = document.getElementById("webcam");
const fileInput = document.getElementById("file-input");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

navigator.mediaDevices
.getUserMedia({ video: true })
.then(stream => {
    video.srcObject = stream;
})
.catch(err => console.error("Webcam initialization failed:", err));
ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

// Button use to choose the cam image
const captureButton = document.getElementById("cam-btn");
captureButton.addEventListener("click", () => {
    document.getElementById("webcam").style.display = "block"; 
    canvas.style.display = "none"; 
});

// Choose image from file input and display in canvas
const fileButton = document.getElementById("chooseFile-btn");
fileButton.addEventListener("click", () => {
    fileInput.click();
});

// upload image from file input and display in canvas
const uploadButton = document.getElementById("upload-btn");
uploadButton.addEventListener("click", () => {
// Convert canvas to a blob and send it for prediction
canvas.toBlob(blob => predict(blob), "image/jpeg");
});

fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (event) {
            const img = new Image();
            img.onload = function () {
                // Get the canvas and context
                const canvas = document.getElementById("canvas");
                const ctx = canvas.getContext("2d");

                // Get original canvas size (defined in HTML or CSS)
                const canvasWidth = canvas.width;
                const canvasHeight = canvas.height;

                // Clear previous drawings
                ctx.clearRect(0, 0, canvasWidth, canvasHeight);

                // Calculate scaling to fit image within canvas while maintaining aspect ratio
                const imgRatio = img.width / img.height;
                const canvasRatio = canvasWidth / canvasHeight;
                let drawWidth, drawHeight, offsetX, offsetY;

                if (imgRatio > canvasRatio) {
                    // Image is wider than canvas: match width
                    drawWidth = canvasWidth;
                    drawHeight = canvasWidth / imgRatio;
                    offsetX = 0;
                    offsetY = (canvasHeight - drawHeight) / 2; // Center vertically
                } else {
                    // Image is taller than canvas: match height
                    drawHeight = canvasHeight;
                    drawWidth = canvasHeight * imgRatio;
                    offsetX = (canvasWidth - drawWidth) / 2; // Center horizontally
                    offsetY = 0;
                }

                // Draw the image scaled and centered on the canvas
                ctx.drawImage(img, offsetX, offsetY, drawWidth, drawHeight);

                // Hide webcam and show canvas
                document.getElementById("webcam").style.display = "none";
                canvas.style.display = "block";
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    } 
});

// Send the image for lesion detection
function predict(file) {
    const loading = document.getElementById("loading");
    const resultTable = document.getElementById("result-table");
    loading.style.display = "block";
    const formData = new FormData();
    console.log("Start detect image");
    formData.append("file", file);

    fetch("/detect", {
        method: "POST",
        body: formData,
    })
        .then((response) => response.json())
        .then((data) => {
            loading.style.display = "none";
            console.log("Detect status:"+ data.lesion_detected);
            if (data.lesion_detected) {
                alert(`Lesion detected with confidence: ${(data.max_confidence * 100).toFixed(2)}%. Proceeding to full prediction.`);
                sendImage(file); // Proceed to prediction if a lesion is detected
            } else {
                alert(`No lesion detected. Please upload a valid image.`);
                loading.style.display = "none";
                resultTable.style.display = "none";
            }
        })
        .catch((error) => {
            loading.style.display = "none";
            alert("Error during lesion detection. Please try again.");
            console.error("Detection error:", error);
        });
}

// Send image to Flask backend for prediction
function sendImage(file) {
    const loading = document.getElementById("loading");
    const resultTable = document.getElementById("result-table");
    loading.style.display = "block";
    resultTable.style.display = "none";

    const formData = new FormData();
    formData.append("file", file);
    console.log("Start send image");

    fetch("/detect", {
        method: "POST",
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            loading.style.display = "none";
            resultTable.style.display = "table";

            //document.getElementById("predicted-class").innerText = data.class;
            document.getElementById("predicted-name").innerText = data.name;
            document.getElementById("description").innerText = data.classDesc;
            document.getElementById("confidence").innerText = `${(data.max_confidence * 100).toFixed(2)}%`;
            console.log("predicted-name:"+ data.name);
            console.log("description:"+ data.classDesc);
            console.log("confidence:"+ data.max_confidence);
        })
        .catch(error => {
            console.error("Error:", error);
            loading.style.display = "none";
            alert("An error occurred. Please try again.");
        });
}
