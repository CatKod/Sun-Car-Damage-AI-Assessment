# Car Damage Dataset Analysis Report

## Dataset Overview
- **Total Images**: 4000
- **Number of Classes**: 6
- **Annotation Format**: COCO -> YOLO
- **Modalities**: Images, Edges, Masks

## Class Distribution
- **dent** (ID: 1)
- **scratch** (ID: 2)
- **crack** (ID: 3)
- **glass shatter** (ID: 4)
- **lamp broken** (ID: 5)
- **tire flat** (ID: 6)

## Metadata Statistics
- **Average Image Size**: 978 x 706
- **Average Categories per Image**: 1.54

### Shooting Angle Distribution
- **side**: 1622 images
- **front**: 1279 images
- **rear**: 649 images
- **inside**: 10 images

### Completeness Distribution
- **partial**: 3920 images
- **complete**: 80 images

## Data Splits
- **Train**: 2816 images, 6211 annotations
- **Val**: 810 images, 1744 annotations
- **Test**: 374 images, 785 annotations
