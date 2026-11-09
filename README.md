
### Project Name-Automated Cup-to-Disc Ratio (CDR) Estimation for Glaucoma Screening Using Classical Image Processing Techniques

### Project Description:This project focuses on developing an automated glaucoma screening system using retinal fundus images and classical image processing techniques. It   identifies the optic disc and optic cup, calculates the Cup-to-Disc Ratio (CDR), and helps indicate potential signs of glaucoma without using deep learning or AI model training.
### Summary - An automated, cost-effective glaucoma screening approach that uses image preprocessing, segmentation, and morphological operations to estimate the CDR from retinal images, reducing the need for manual measurements.

### Course concepts used - 
1. Unit 1 – Digital Image Fundamentals
 Image Acquisition
 Pixel Processing
 Region of Interest (ROI)
 Edge and Boundary Detection
2. Unit 2 – Image Transforms
 Discrete Fourier Transform (DFT)
 Frequency Domain Filtering
 Singular Value Decomposition (SVD)
 Noise Reduction
3. Unit 3 – Image Enhancement
 Histogram Equalization
 CLAHE (Contrast Limited Adaptive Histogram Equalization)
 Spatial Filtering
 Low-Pass Filtering
 Blood Vessel Enhancement and Removal
4.Unit 4 – Morphological Image Processing
  Erosion and Dilation
  Opening and Closing
  Shape Refinement and Segmentation
  Color and Pseudocolor Processing
   
 ### Additional concepts used -
1. -OTSU thresholding
2. -Canny Edge Detection
3. -Cup-to-Disc Ratio (CDR) Calculation
   
### Dataset - 
Drishti-GS retinal fundus image dataset

### Novelty - 
1. Clinical Validation: Compares automatically calculated CDR values with measurements provided by multiple ophthalmologists.

2. Statistical Agreement Analysis: Uses Bland-Altman analysis to measure the agreement and differences between computer-generated and doctor-measured CDR values.

3. Comparative Segmentation: Implements and compares 2–3 different segmentation techniques to identify the most effective approach.

4. Inter-Observer Variability Analysis: Evaluates the system's accuracy against the variations in CDR measurements between different doctors.

5. Beyond Segmentation: Focuses not only on identifying the optic disc and cup but also on validating the accuracy of the calculated CDR values against clinical measurements.
   
### Contributors:
1. C S Anvitha-PES1UG24EC249
2. Bhagyashri K S-PES1UG24EC048
3. Dhanyashree D-PES1UG24EC340

### Steps:
1. Clone Repository
git clone https://github.com/DIP-2026-MINI-PROJECT/Automated-Cup-to-Disc-Ratio-CDR-Estimation-for-Glaucoma-Screening-.git

2. Requirements
MATLAB with the Image Processing Toolbox R2025b

3. Run the Code
    cdr_estimation.m

### Outputs:
* Important intermediate steps
* Final output images 

### References:
1. -"Identifying the Edges of the Optic Cup and the Optic Disc..." (Sensors, PMC)
2. -"Optic Disc and Cup Segmentation... Locally Statistical Active Contour Model with Structure Prior" (PMC)
3. -"A novel optic disc and optic cup segmentation technique..." (ScienceDirect, 2021)
4. -"Performance Analysis Of Morphological Operations in CPU and GPU..." (ResearchGate)
5. -""Efficient parallel implementation of morphological operation on GPU and FPGA" (IEEE Xplore)
   
### Limitations and Future Work:
Limitations-
1.-Blood Vessel Interference: Blood vessels crossing the optic cup can make accurate boundary detection difficult.
2.-Dataset Limitation: Performance may vary when tested on retinal images from different datasets or imaging devices.
3.-Manual Validation Dependency: Reliable performance evaluation requires reference measurements from ophthalmologists.
Future Work-
1.-Better Vessel Removal: Develop more effective blood vessel detection and removal methods.
2.-Multi-Dataset Testing: Validate the system using additional datasets such as RIM-ONE and REFUGE.
3.-Clinical Validation: Improve reliability through comparison with multiple ophthalmologists' measurements.
4.-Statistical Analysis: Extend Bland-Altman analysis to evaluate agreement and measurement consistency.
