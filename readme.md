
# 🚀 Welcome to the Soil Image Classification (Local Setup Guide)

## 🛠️ Setup Instructions

### Step 1: Project Structure

1. Create a folder named `root`.
2. Open the folder in **VS Code**.
3. Clone the repository:

```bash
git clone --branch main https://github.com/Keshav-CUJ/Soil_Classification_annam.git 
```

---

### Step 2: Set Up Virtual Environment

In the terminal (PowerShell), ensure you’re in the `root` folder and run:

```powershell
python3.9 -m venv Newvenv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\Newvenv\Scripts\Activate.ps1
```

---

### Step 3: Install Python Dependencies

Make sure `requirements.txt` file is in root folder. Then, run:

```bash
pip install -r requirements.txt
```

---

### Step 4: Setup Dataset

**Option 1:** Download both datasets manually and place the files (`train`, `test`, and `test_ids.csv`) into the `data` folder inside `Challenge-1`.

**Option 2:** Use Kaggle CLI for automatic download via `download.sh`.

```bash
cd Challenge-1/data
```

**Set up Kaggle API Credentials:**

1. Go to your Kaggle account: https://www.kaggle.com/account
2. Scroll to **API** section → Click **"Create New API Token"**
3. It will download a file named `kaggle.json`

Then, in your terminal:

```bash
mkdir -p ~/.kaggle
cp /path/to/your/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

To run the dataset download script:

```bash
chmod +x download_kaggle_dataset.sh
./download_kaggle_dataset.sh
```

### ✅ Final Folder Structure:

<pre>
Root/
└── Challenge-1/
    ├── data/
    │   ├── download.sh
    │   ├── data/
    │        ├── train
    |        ├── test 
    │        ├── train_ids.csv
    |        ├── test_ids.csv 
    ├── notebook/
    │   ├── inference.ipynb
    │   └── training.ipynb
    ├── src/
    │   └── preprocessing.py
    ├── trained_model/
    │   └── .pth or .joblib
</pre>

---

## 🧠 Model Training & Inference for both challenges

### Step 5: Training the Model

1. Delete the previously saved models from `trained_model` folder
2. Run `preprocessing.py` from `src` folder.
3. Visit `training.ipynb` in the `notebooks` folder and run the notebook.
4. Trained model will be saved into the `trained_model` folder.

### Step 6: Run Inference on New Test Data

1. Replace `test_ids.csv` and `test` folder with the new test data inside `data` folder.
2. Visit `inference.ipynb` in the `notebooks` folder and run the notebook.
3. The result CSV (with labels) will be saved into the `notebooks` folder.

---


