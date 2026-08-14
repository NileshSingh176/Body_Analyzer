Body Analyzer

An AI-powered sports movement analysis platform that analyzes 8 different exercises using MediaPipe Pose and YOLOv8. The application provides exercise form analysis, repetition counting, performance metrics, and annotated output through a Flask web interface.

⸻

Features

* Human pose estimation using MediaPipe
* Person detection using YOLOv8
* Exercise-specific movement analysis
* Automatic repetition counting
* Form evaluation and scoring
* Joint angle and movement analysis
* Annotated video/image output
* JSON analysis reports
* SQLite-based result storage
* Flask web interface

⸻

Supported Exercises

Exercise	Analysis
Push-ups	Elbow angle, back alignment, reps
Squats	Knee/hip angles, depth, reps
Single-leg Squats	Knee angle, pelvic movement, reps
Broad Jump	Jump distance and landing
Walking Lunges	Knee depth, trunk angle, reps
Squat with Stick	Squat depth, wrist position
Vertical Jump	Jump height and flight time
20m Sprint	Speed and movement metrics

⸻

How It Works

Video / Image
      ↓
Pose & Person Detection
      ↓
Body Landmark Extraction
      ↓
Joint Angle & Movement Analysis
      ↓
Exercise-specific Evaluation
      ↓
Rep Count + Form Score
      ↓
Analysis Result & Annotated Output

⸻

Tech Stack

* Python
* Flask
* MediaPipe Pose
* YOLOv8
* OpenCV
* NumPy
* SQLite
* HTML / CSS / JavaScript

⸻

Project Structure

Body_Analyzer/
│
├── modules/
│   ├── app.py
│   ├── database.py
│   ├── utils.py
│   ├── module_pushups.py
│   ├── module_squat.py
│   ├── module_single_leg_squat.py
│   ├── module_broad_jump.py
│   ├── module_walking_lunges.py
│   ├── module_squat_with_stick.py
│   ├── module_vertical_jump.py
│   └── module_speed_20m.py
│
├── models/
├── static/
├── templates/
├── requirements.txt
├── .gitignore
└── README.md

⸻

Installation

Clone the repository:

git clone https://github.com/NileshSingh176/Body_Analyzer.git
cd Body_Analyzer

Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

⸻

Run the Application

python3 modules/app.py

Then open:

http://localhost:5000

⸻

Input & Output

The application accepts common video and image formats such as:

.mp4  .avi  .mov  .webm
.jpg  .jpeg  .png  .bmp

The system generates exercise analysis results including:

* Repetition count
* Form score
* Movement metrics
* Detected issues
* Annotated output

⸻

Screenshots

Add screenshots or demo images here to showcase the application.

![Body Analyzer](path/to/screenshot.png)

⸻

Limitations

Analysis accuracy can be affected by:

* Camera angle
* Poor lighting
* Occluded body parts
* Low video quality
* Multiple people in the frame
* Incorrect camera positioning

⸻

Author

Nilesh Singh

AI/ML Engineer | Computer Vision | Machine Learning | Python

GitHub: NileshSingh176
