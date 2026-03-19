import requests
import sys
import os

def test_api(image_path, api_url):
    print(f"Testing API: {api_url}")
    print(f"Image used: {image_path}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"ERROR: Image file '{image_path}' not found.")
        return

    # Set up the multipart file upload
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
        
        try:
            # Send POST request to /predict
            response = requests.post(f"{api_url.rstrip('/')}/predict", files=files, timeout=30)
            
            # Print results
            if response.status_code == 200:
                print("\nSUCCESS! Prediction Result:")
                print(response.json())
            else:
                print(f"\nFAILED! Status Code: {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"\nERROR: Could not connect to API: {str(e)}")

if __name__ == "__main__":
    # USER: REPLACE THIS WITH YOUR CLOUD RUN URL
    # Example: "https://resnet-api-xxxxx-as.a.run.app"
    SERVICE_URL = "https://resnet-api-297419987783.asia-south1.run.app"
    
    # Path to sample image in your workspace
    SAMPLE_IMAGE = "path/to/a/paddy_seed_image.jpg"
    
    if SERVICE_URL == "INSERT_YOUR_CLOUD_RUN_URL_HERE":
        print("Please replace 'SERVICE_URL' in this script with your Cloud Run URL from the GitHub Actions logs.")
    else:
        test_api(SAMPLE_IMAGE, SERVICE_URL)
