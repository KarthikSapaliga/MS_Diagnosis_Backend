# MS Early Diagnosis - Backend

```
backend
├── app.py
├── env/                      # virtual environment
├── models/
│   ├── mri_model/
│   │   ├── cat_model.cbm
│   │   ├── meta.json
│   │   ├── rf_meta.pkl
│   │   └── xgb_model.pkl
│   └── oct_model/
│       └── oct_model.pth
├── pkg/
│   ├── mri_inference.py
│   └── oct_inference.py
├── requirements.txt
└── test/                    # test dataset
    ├── mri
    └── oct
```