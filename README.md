# Cat Feeder Cam

Cat Feeder Cam is a Raspberry Pi project designed for students and educators. It uses computer vision to detect a cat and triggers an actuator, such as a servo motor, to dispense food. Ideal for STEM activities, classroom demonstrations, and DIY hobbyists learning about AI and electronics.

## What is this project?

This is a beginner-friendly project built with Python, OpenCV, and GPIO. It includes:

- Real-time object detection using a pre-trained MobileNetSSD model
- Camera input from either a USB webcam or the official Raspberry Pi camera module
- Servo activation via GPIO when a target (cat or person) is detected
- Optional 3D-printable parts for a basic feeder mechanism
- Clean and modifiable code for experimentation

## Features

- Compatible with Raspberry Pi 4 or newer
- Supports both USB and DSI cameras
- Lightweight and optimized for classroom performance
- Includes STL files for optional 3D-printed feeder parts
- Written for students, with clear structure and instructions

## Requirements

| Item             | Description                          |
|------------------|--------------------------------------|
| Raspberry Pi     | Model 4 recommended                  |
| Camera           | USB webcam or Raspberry Pi camera    |
| Servo motor      | SG90 or equivalent                   |
| Jumper wires     | Female-to-male, for GPIO connection  |
| Python           | Python 3.x with OpenCV support       |

## Getting Started

1. Clone the repository:

    ```bash
    git clone https://github.com/firdausaris/cat-feeder.git
    cd cat-feeder
    ```

2. Install dependencies:

    ```bash
    sudo apt update
    sudo apt install python3-opencv python3-gpiozero -y
    ```

3. Run the program:

    ```bash
    python3 cat_feeder.py
    ```

## Educational Value

This project introduces students to:

- Object detection with OpenCV
- Using GPIO pins on the Raspberry Pi
- Practical use of machine learning models
- Simple automation and actuator control
- Basic 3D printing (if hardware components are used)

## Suggested Extensions

- Add a notification system (e.g., Telegram alert)
- Build a simple web interface
- Use a light or sound indicator when detection occurs
- Add a delay or cooldown to prevent repeat triggers
- Swap the camera source dynamically in code

## License

This project is licensed under the MIT License.  
The included MobileNetSSD model is based on work by Chuanqi Wu (BSD 2-Clause License).

## Credits & Model License

This project includes the MobileNetSSD model, originally published by Chuanqi Wu:
- https://github.com/chuanqi305/MobileNet-SSD

The model is licensed under the BSD 2-Clause License (see `model/LICENSE`).

## Attribution

Project developed by Firdaus Aris (2025).  
Designed as an open educational tool for learning computer vision, electronics, and automation.  
Research profile: [https://orcid.org/0009-0006-0458-9782](https://orcid.org/0009-0006-0458-9782)




