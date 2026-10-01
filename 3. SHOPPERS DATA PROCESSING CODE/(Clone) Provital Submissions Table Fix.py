# Databricks notebook source
# MAGIC %sql
# MAGIC SELECT * FROM `workspace`.`provital`.`_1bxw132rkxiqhshpcvzmmbwhf8tyb7brqfcoye7ep1da`;

# COMMAND ----------

# DBTITLE 1,Fix: Create Submissions Table with Correct Column Names
# MAGIC %md
# MAGIC # Fix: Create Submissions Table with Correct Column Names
# MAGIC
# MAGIC The original streaming table (`workspace.provital._1bxw...`) shows nulls in most columns because the Google Spreadsheet headers use **underscores** (e.g., `Lead_ID`, `First_Name`) while the table schema was defined with **spaces and different names** (e.g., `Lead ID`, `First Name`, `Province/State`). Only `Country` and `City` matched exactly and had data.
# MAGIC
# MAGIC This cell creates a new Delta table `workspace.provital.submissions` with column names matching the spreadsheet exactly, populated from the current spreadsheet data.

# COMMAND ----------

# DBTITLE 1,Create submissions table with correct column names
# Column names matching the Google Spreadsheet headers exactly
headers = [
    "Lead_ID", "Submitted_At", "First_Name", "Country", "Province",
    "City", "Product_Interests", "Other_Interests", "How_They_Found_Us",
    "Email", "MobileWhatsApp", "Marketing_Consent", "Source_Page",
    "UTM_Source", "UTM_Medium", "UTM_Campaign", "Race_Group",
    "Gender", "Discovery_Source"
]

# Data rows from the Google Spreadsheet ("Submissions" tab)
# Each row may be shorter than 19 columns; we pad with None and convert empty strings to None
raw_data = [
    ["PV-MUMWVS26-56IL", "2026-09-29T16:49:26.478Z", "Proh", "South Africa", "Western Cape", "Cape Town", "Herbs & herbal teas", "hair products", "Google/Search", "prohzindove@gmail.com", "786520728", "Yes", "https://lovable.dev/"],
    ["PV-MUMZP0SA-2C75", "2026-09-29T18:08:10.042Z", "Nthabiseng Rakuoane", "South Africa", "Gauteng", "Midrand", "", "", "Other", "rakuoanenthabiseng17@gmail.com", "", "Yes", "https://www.google.com/"],
    ["PV-MUMZQ8HJ-23WV", "2026-09-29T18:09:06.679Z", "Mpho Lesunyane", "South Africa", "Gauteng", "Pretoria", "Natural skincare", "Not at the moment", "WhatsApp", "mlesunyane@gmail.com", "825739577", "Yes", "https://www.google.com/"],
    ["PV-MUMZQUH7-1DZ7", "2026-09-29T18:09:35.179Z", "Busisiwe Khoza", "South Africa", "Gauteng", "Midrand", "Natural skincare, Herbs & herbal teas, Honey & bee products, Dried berries & superfoods", "Bloating remedies", "Friend or family", "khozabusisiwe950@gmail.com", "", "Yes", "https://www.google.com/"],
    ["PV-MUMZRXYM-SWUH", "2026-09-29T18:10:26.350Z", "Rofhiwa", "South Africa", "Eastern Cape", "East London", "Natural skincare", "Mpensu cheap cheap", "Instagram", "rofhiwanemukula98@gmail.com", "787372893", "Yes", "https://form-to-sheet-friend.lovable.app/"],
    ["PV-MUMZWWAA-ZUJC", "2026-09-29T18:14:17.458Z", "Rogs Rogie", "South Africa", "Free State", "Welkom", "Wellness lifestyle, Natural skincare", "Good health", "TikTok", "rogs3@gmail.com", "723346678", "Yes", "https://www.google.com/"],
    ["PV-MUMZYO5F-Y0QH", "2026-09-29T18:15:40.227Z", "Gugu", "South Africa", "Mpumalanga", "Emalahleni", "Dried berries & superfoods", "", "Other", "gugu@gmail.com", "751254785", "Yes", "https://www.google.com/url?sa=j&url=https%3A%2F%2Fform-to-sheet-friend.lovable.app%2F&uct=1756281155&usg=2mBDf-3WHj9w6mUFtIaOKrKI-5s.&opi=76390225&source=meet"],
    ["PV-MUN3MM2N-ACQZ", "2026-09-29T19:58:16.127Z", "Malils", "South Africa", "Gauteng", "Johannesburg", "Herbs & herbal teas, Natural skincare", "", "Friend or family", "machabalilly@gmail.com", "734757783", "Yes", "https://www.google.com/"],
    ["PV-MUNU2MV3-4Z9D", "2026-09-30T08:18:33.663Z", "Nomvano", "South Africa", "", "Kimberley ", "Wellness lifestyle", "", "WhatsApp", "princessmalatha@gmail.com", "735056508", "No", "https://form-to-sheet-friend.lovable.app/"],
]

# Pad short rows to full 19 columns and convert empty strings to None
padded_data = []
for row in raw_data:
    padded = list(row) + [None] * (len(headers) - len(row))
    padded = [None if v == "" else v for v in padded]
    padded_data.append(padded)

# Define explicit schema (all strings; Submitted_At will be cast to timestamp after)
from pyspark.sql.types import StructType, StructField, StringType

schema = StructType([StructField(col, StringType(), True) for col in headers])

# Create Spark DataFrame with correct column names and explicit schema
df = spark.createDataFrame(padded_data, schema)

# Cast Submitted_At from string to timestamp
df = df.withColumn("Submitted_At", df["Submitted_At"].cast("timestamp"))

# Write as a Delta table in the workspace.provital schema
df.write.format("delta").mode("overwrite").saveAsTable("workspace.provital.submissions")

print(f"✅ Created workspace.provital.submissions: {df.count()} rows, {len(headers)} columns")
display(df)

# COMMAND ----------

# DBTITLE 1,Verify new submissions table
# MAGIC %sql --name verify_submissions
# MAGIC SELECT * FROM workspace.provital.submissions ORDER BY Submitted_At DESC;