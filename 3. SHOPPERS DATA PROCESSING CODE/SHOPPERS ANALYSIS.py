# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # ONLINE SHOPPERS DATA ANALYSIS

# COMMAND ----------

# MAGIC %md
# MAGIC # Introduction
# MAGIC The Head of Operations wants to understand our online store performance from 1 January 2024 to 30 June 2026.
# MAGIC
# MAGIC Four datasets were provided for the analysis, namely, Customers, Orders, Payments, and Products. The datasets will first be loaded and converted into Pandas DataFrames, then inspected and cleaned separately to ensure data quality before being joined. The Orders dataset will serve as the main dataset and will be linked to Customers using CustomerID, Products using ProductID, and Payments using OrderID. After joining the datasets, feature engineering will be performed to create additional variables needed for the analysis. The final dataset will then be analysed to answer the following business questions and provide insights into the online store's performance:
# MAGIC
# MAGIC * Revenue & Orders: How much revenue did the shop make, from how many orders? What is the average order value?
# MAGIC * Revenue Trend: Is revenue growing, shrinking or flat month by month? Are there any seasonal peaks?
# MAGIC * Product Performance: Which products and categories bring in the most revenue? Which sell the most units? Are these the same?
# MAGIC * Customer Value: Which cities and customer segments (New, Regular, VIP) are most valuable?
# MAGIC * Order & Payment Status: What share of orders are cancelled or returned? What share of payments fail? Do some payment methods fail more often?
# MAGIC * Discount Impact: Do bigger discounts lead to bigger orders, or just lower revenue?
# MAGIC
# MAGIC The analysis will use totals, averages, counts and percentages, and data grouping where appropriate.

# COMMAND ----------

# MAGIC %md
# MAGIC # Importing libraries

# COMMAND ----------

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Customers dataset 

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Customers data ingestion
# MAGIC This is where we load customers data into our python notebook for processing. The dataset is stored as a spark table in databricks hence we convert it to a pandas dataframe before using it in the notebook.

# COMMAND ----------

customers =spark.table("shoppers.data.customers").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Customers data inspection
# MAGIC In this section the dataset is inspected to understand its structures, check for any errors such as duplicates and missing values and verify data quality. 

# COMMAND ----------

# checking the first few columns of the dataset to have an idea of what it contains.
customers.head()

# COMMAND ----------

# checking the number of rows and columns in the dataset
customers.shape

# COMMAND ----------

# checking for a general summary of the dataset
customers.info()

# COMMAND ----------

# MAGIC %md
# MAGIC Age and City columns do not have a total of 10 000 entries which is a clear indication of missing values. These will be dealt with in the data cleaning section, Age will be filled in with a median value and City will be replaced with Unknown. In addition, Age was captured as a float but the age entries are whole numbers, it will be transformed to an integer. Lastly, SignupDate was captured as an object, it will be transformed to a datetime.

# COMMAND ----------

# checking to confirm the missing values and their count
customers.isna().sum()

# COMMAND ----------

# checking for rows where both Age and City have missing values using the & filtering function.
both_missing_values = customers[customers["Age"].isnull() & customers["City"].isnull()]
display(both_missing_values)

# COMMAND ----------

# MAGIC %md
# MAGIC Only 4 rows have missing values in both Age and City columns

# COMMAND ----------

# checking for any duplicated rows
customers.duplicated().sum()

# COMMAND ----------

#counting the number of cities in the dataset
customers["City"].nunique()

# COMMAND ----------

#checking the number of customers in each unique city in the dataset
customers["City"].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC It shows that there are entry errors where Tehran was aalos entered as tehran with a small letter t and Mashhad was also entered as Mashad with a single h. These will be standardised to the entry with more values (Tehran and Mashhad)

# COMMAND ----------

# checking the number of unique customer segments
customers["CustomerSegment"].nunique()

# COMMAND ----------

# checking the number of customers in each unique customer segment
customers["CustomerSegment"].value_counts()

# COMMAND ----------

# checking the summaries of numerical data
customers.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC Minimum Age is 18 and maximum is 65, there is a wide gap hence AgeGroup buckets to categorise people of similar age will be created. Mean age is 41.3, however, it will not be used to fill in the missing values to avoid skewness due to the large gap between the minimum and maximum age groups, the median 41 will be used. 

# COMMAND ----------

# checking for the first customer signup date 
customers["SignupDate"].min()

# COMMAND ----------

# checking for the last customer signup date 
customers["SignupDate"].max()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2.1 Customers data inspection summary
# MAGIC Customers dataset contains 10,000 rows and 5 columns. An initial check showed missing values in Age and City columns, these will be handled in the data cleaning section. It also showed that SignupDate was saved as an object instead of a date, it will be transformed under data cleaning section. Further inspection found 180 missing values in Age and 119 in City columns. I also checked whether both values were missing from the same rows and it shows that only 4 rows had the same missing values. The dataset contains 12 cities, however city Terhan is also entered as terhan and Mashhad also entered as Mashad. These will be standardised in the data cleaning section. There are three customer segments. Regular is the largest, with 5,563 customers, and VIP is the smallest, with 975. A summary of the numerical values showed a wide age range, so it may be useful to create age groups. The signup dates run from 1 January 2023 to 30 December 2025, covering almost three years.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 Customers data cleaning
# MAGIC Some data quality issues were found in the dataset such as missing values in Age and City column. There are two entries in the city column that will be standardisd. Age and SignupDate are saved as a float and an object respectively, instead of an integer and a date. All be handled in this section. 

# COMMAND ----------

# MAGIC %md
# MAGIC **Cleaning City column**

# COMMAND ----------

# standardising city entries that were wrongly entered
customers["City"]=customers["City"].replace({'tehran': 'Tehran','Mashad': 'Mashhad'})

# COMMAND ----------

# checking to confirm if the city entries are standardised
customers["City"].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC Wrongly captured cities were standardised, we now have a total of 10 cities instead of the initial 12.

# COMMAND ----------

# handling missing values in City column. Missing values will be replaced with an Unknown
customers["City"] = customers["City"].fillna("Unknown")

# COMMAND ----------

# MAGIC %md
# MAGIC **Cleaning Age column**

# COMMAND ----------

# handling missing values in Age column. Since Age contains numerical values, we will add a median value to replace the missing values
median_age=customers["Age"].median()

customers["Age"] = customers["Age"].fillna(median_age)

# COMMAND ----------

# transforming Age column from a float to an integer
customers["Age"]= customers["Age"].astype(int)

# COMMAND ----------

# MAGIC %md
# MAGIC Missing values in Age column were replaced with the median age of 41. Median age was chosen because it represents a typical age in the dataset and allows the Age column to remain as whole numbers without being influenced by unusually high or low values. Age was then transfered from a float to an integer. 

# COMMAND ----------

# MAGIC %md
# MAGIC **Cleaning date column**

# COMMAND ----------

# transforming signup date from object to date function
customers["SignupDate"] = pd.to_datetime(customers["SignupDate"])

# COMMAND ----------

# checking to confirm if the cleaning was effected
customers.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3.1 Customers data cleaning summary
# MAGIC Data cleaning was done to improve data quality. Missing values in City were replaced with “Unknown”. While missing values in Age column were filled with the median age. Age column was also transformed to an integer. Five customer age groups were then created. SignupDate was converted from an object to a date. A final check confirmed that all changes were applied. 

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Orders dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Orders data ingestion
# MAGIC This is where we load our orders data into our python notebook for processing. Our dataset is stored as a spark table in databricks hence we convert it to a pandas dataframe before using it in our notebook.

# COMMAND ----------

orders =spark.table("shoppers.data.orders").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Orders data inspection
# MAGIC In this section the dataset is inspected to understand its structures, check for any errors such as duplicates and missing values and verify data quality. 

# COMMAND ----------

# checking the first few rows in the dataset
orders.head()

# COMMAND ----------

# checking the number of rows and columns in the dataset
orders.shape

# COMMAND ----------

# checking for data summary
orders.info()

# COMMAND ----------

# MAGIC %md
# MAGIC A number of columns have missing values (OrderDate, Quantity, Discount, and Paymentmethod), these will be handled in the data cleaning section once duplicates are checked and handled. Quantity is saved as a float, it will be transformed to a whole number since the products being sold are whole. 

# COMMAND ----------

# seperately checking for the number of missing values per column 
orders.isna().sum()

# COMMAND ----------

# checking to see if the missing values are in the same rows
missing_values = orders[orders["OrderDate"].isnull() & orders["Quantity"].isnull() & orders["Discount"].isnull() & orders["PaymentMethod"].isnull()]

display(f"Number of rows with all four values missing: {len(missing_values)}")
if not missing_values.empty:
    display(missing_values)

# COMMAND ----------

# MAGIC %md
# MAGIC Missing values were checked seperately to confirm the total for each column the least is Orderdate with 35 and the biggest number is in the PaymentMethod column with 432 missing values. The missing values were further checked to see if they are in the same rows but none were in the same rows.

# COMMAND ----------

#checking for duplicates
orders.duplicated(). sum()

# COMMAND ----------

# MAGIC %md
# MAGIC It shows that there are 120 duplicates, these will be dropped in the data cleaning section and should reduce the size of the entries to 50 000.

# COMMAND ----------

# checking for the value counts for each unique PaymentMethod
orders["PaymentMethod"].value_counts()

# COMMAND ----------

# further confirmation of the unique payment methods since the null entries did not show in the previous entry
orders["PaymentMethod"].unique()

# COMMAND ----------

# MAGIC %md
# MAGIC The output further confirms the None entry in PaymentMethod column

# COMMAND ----------

# checking for unique status
orders["Status"].value_counts()

# COMMAND ----------

# checking for numerical column summaries
orders.describe()

# COMMAND ----------

# checking quantity values and their frequencies
orders["Quantity"].value_counts(dropna=False).sort_index()

# COMMAND ----------

# MAGIC %md
# MAGIC It shows that there are 19 negative quantity and 6, 0 quantities. This needs further investigation to check the status of those rows.

# COMMAND ----------

# filtering to view invalid and missing quantities using OR function
quantity_quiries=orders[orders['Quantity'].isna() | (orders['Quantity'] <= 0)]
display (quantity_quiries)

# COMMAND ----------

# checking the status pattern of rows with quiries
quantity_quiries['Status'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC Out of 105 entries with anormalies, 96 show that the sale was completed and only 8 were cancelled and 1 returned. We could not correctly identify the problem so since the affected rows are very few, these will be dropped when we clean the data.

# COMMAND ----------

# checking discount values and their frequencies
orders["Discount"].value_counts(dropna=False).sort_index()

# COMMAND ----------

# MAGIC %md
# MAGIC It shows that a large number of products was sold on 0 discount and in general the largest number was bewteen 0 and 0.10. The discounts range between 0.0 and 0.30 and an assumption is made that the large discounts were on selected items based on the quantities sold. The null values need further investigation to establish if they are really missing or zeros were recorded as missing and also to check the order status.

# COMMAND ----------

# checking for orders with missing discount
missing_discount = orders[orders['Discount'].isna()]

display(missing_discount)



# COMMAND ----------

missing_discount['Status'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC Missing values in discount column show that they are really missing values and not zeros. Out of the 220 missing values, 207 show that the orders were completed, 11 were canceled and 3 were returned. The large number of completed orders shows that the values are indeed missing values. They will be dropped when cleaning data.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2.1 Orders data inspection summary
# MAGIC The first step in the data inspection section was to check a few rows to see what the data contins. This was followed by checking the number of rows and columns in the dataset, it contains 50120 rows and 8 columns. The data summary overview was then checked to have a high level understanding dataset contents. It shows that the order date column was wrongly saved as an object instead of saving it as a date. A transformation of this column will be needed. In addition, there are four incomplete columns order date, quantity, discount and payment method. Total missing values are 35 for order date, 80 for quantity, 221 in discount and 452 in payment method column. There are also 120 duplicated rows in the dataset and they will be handled in the next section.There are four uniqe payment methods and three statuses. Quantity and Discount columns were further investigated and since they hold a very small amount, they will be dropped.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Orders data cleaning
# MAGIC In the previous section we realised that there are missing values in four columns and duplicated rows. Order date is also saved as an object instead of a date. These will be handled in this section.

# COMMAND ----------

# MAGIC %md
# MAGIC **Removing duplicates**

# COMMAND ----------

# handling duplicated rows
orders = orders.drop_duplicates()

# COMMAND ----------

#confirming if the duplicates were dropped
orders.duplicated().sum()

# COMMAND ----------

# checking the number of missing values after dropping duplicates
orders.isna().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC Duplicated rows were successfully removed and further checks show that only 3 rows of the missing data rows were duplicated

# COMMAND ----------

# MAGIC %md
# MAGIC **Handling missing values**

# COMMAND ----------

# handling missing values- dropping missing values in columns OrderDate, Quantity and Discount and reset index after dropping 
orders = orders.dropna(subset=["OrderDate", "Quantity", "Discount"])

orders = orders.reset_index(drop=True)

# COMMAND ----------

# handling missing values- filling missing values in column PaymentMethod, Quantity and Discount and reset index after dropping 
orders["PaymentMethod"]= orders["PaymentMethod"].fillna("Unknown")

# COMMAND ----------

orders.isna().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC Missing values were handled by removing them in the columns were we could not find a perfect replacement due to missing information and the small number of misssing values. However in the PaymentMethod we filled the missing values with Unknown. 

# COMMAND ----------

# MAGIC %md
# MAGIC **Standardisation of Quantity and Date columns**

# COMMAND ----------

# changing Qunatity column from float to integer
orders["Quantity"] = orders["Quantity"].astype(int)

# COMMAND ----------

# transforming date from object to date
orders["OrderDate"] = pd.to_datetime(orders["OrderDate"])

# COMMAND ----------

# confirmation
orders.dtypes

# COMMAND ----------

#checking the new dataset size after data cleaning
orders.shape

# COMMAND ----------

#checking the minimum order date
orders["OrderDate"].min()

# COMMAND ----------

# checking the last order date
orders["OrderDate"].max()

# COMMAND ----------

# MAGIC %md
# MAGIC Transformations were done and Quantity is now correctly saved as an integer and date is now correctly saved as date column. The new rows afte cleaning are now 49665. Further checks on start and end date of the dataset was then done since it could not be done before cleaning the data.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3.1 Orders data cleaning summary
# MAGIC Data cleaning was successfuly effected starting with dropping duplicates. This first step helped me check in case the duplicates were part of the missing values. Missing values were then dealt with, some were dropped (OrderDate, Quantity, Discount) and some were filled in (PaymentMethod). Transformations were also done on  Quantity and OrderDate.

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Payments Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Payments data ingestion
# MAGIC In this section, we are importing the dataset into python notebook. We further convert the dataset to a pandas dataframe for further processing.

# COMMAND ----------

payments =spark.table("shoppers.data.payments").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Payments data inspection
# MAGIC In this section the payments dataset is inspected to understand its structures, check for any errors such as duplicates and missing values and verify data quality. 

# COMMAND ----------

# checking the first few columns
payments.head()

# COMMAND ----------

# checking the number of rows and columns in payments
payments.shape

# COMMAND ----------

# checking for general summaries
payments.info()

# COMMAND ----------

# MAGIC %md
# MAGIC It shows that the total number of rows in this dataset are 50 000, PaymentDate however shows that it is incomplete with only 49965 rows. in addition, it is also saved as an object instead of a date, it will be transformed later.

# COMMAND ----------

# confirming the missing value count
payments.isna().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC The confirmed number of missing values is 35, these will be deleted since it's a very small number.

# COMMAND ----------

# checking for duplicates
payments.duplicated().sum()

# COMMAND ----------

# checking for the summary of numerical columns
payments.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC The numerical values in this dataset are for identification purposes and no further processing will be required.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2.1 Data inspection summary
# MAGIC The datasets' first few raws were inpsected for an oversight of it's contents. The size of the dataset is 50 000 rows and 4 columns. a general summary also shows that payment date is saved as an object and has missing values and they will be delt with in the next section. The number of missing values was confirmed to be 35 and only in the date column. There are no duplicate rows in the dataset. The numerical summaries will no be useful in our further processing since they are just for identification purposes.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Payments data cleaning
# MAGIC We noticed that PaymentDate has missing values and is wrongly stored as an object instead of a  date. These anormalies will be handled in this section.

# COMMAND ----------

# MAGIC %md
# MAGIC **Handling missing values**

# COMMAND ----------

# dropping missing valuees in PaymentDate column
payments = payments.dropna(subset=["PaymentDate"])

# COMMAND ----------

# MAGIC %md
# MAGIC **Transforming PyamentDate from object to date**

# COMMAND ----------

# changing PaymentDate from object to date
payments["PaymentDate"] = pd.to_datetime(payments["PaymentDate"])

# COMMAND ----------

# confirming
payments.info()

payments.isnull().sum()

# COMMAND ----------

# checking start date of payments
payments["PaymentDate"].min()

# COMMAND ----------

# checking last date of payments
payments["PaymentDate"].max()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3.1 Payments data cleaning summary
# MAGIC Missing values were dropped and PaymentDate was transformed from object to date format. The changes were confirmed and the new number of rows is 49965. PaymentDate successfully transformed to date.

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Products dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Products dataset ingestion

# COMMAND ----------

products =spark.table("shoppers.data.products").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Products dataset inspection

# COMMAND ----------

# checking the first five rows of the data
products.head()

# COMMAND ----------

# checking the data size
products.shape

# COMMAND ----------

# getting a summary of the dataset
products.info()

# COMMAND ----------

# MAGIC %md
# MAGIC It shoes that products dataset has only 20 rows and 4 columns. It also shows that there are no null values and all columns are stored in correct data types

# COMMAND ----------

# checking the unique product names that are in the dataset
products["ProductName"].unique()

# COMMAND ----------

# counting the number of unique productnames in the dataset
products["ProductName"].nunique()

# COMMAND ----------

# counting the number of each unique product name
products["ProductName"].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC There are 20 unique product names with each having one count and all entered correctly.

# COMMAND ----------

# checking the number of unique entries in the category column
products["Category"].unique()

# COMMAND ----------

# counting the number of unique category entries
products["Category"].nunique()

# COMMAND ----------

# checking value counts in Category column
products["Category"].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC It shows that there are 6 product categories with the highest being electronics with 9 categories. Gaming might sometimes fall under the main electronics but since the business categorises it that way and did not specify in the case study description, it will not be changed.

# COMMAND ----------

# checking the summaries of the numerical columns
products.describe()

# COMMAND ----------

# sorting products from highest to lowest price
products.sort_values("UnitPrice", ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC It shows that minimum UnitPrice is 7 and maximum is 260. Further checks were done to have an understanding of the prices of each product. It does not look like there are any outliers.

# COMMAND ----------

# checking the rows for any duplicates
products.duplicated().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2.1 Products data cleaning summary
# MAGIC Data inspection confirms that products dataset is clean and no further cleaning is required. 

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Joining the four tables
# MAGIC In this section we are going to join the four datasets and create one combined dataset for analysis. The Orders dataset will be used as the main table because it contains the transaction records. The datasets will be joined using the following common keys:
# MAGIC
# MAGIC * Customers will joined to Orders using CustomerID.
# MAGIC
# MAGIC * Products will be joined to Orders using ProductID.
# MAGIC
# MAGIC * Payments will be joined to Orders using OrderID.
# MAGIC
# MAGIC A left join will be used to retain all records from the Orders dataset while adding the relevant customer, product and payment information. 

# COMMAND ----------

# joining Orders to Customers
shoppers_df = orders.merge(customers,on="CustomerID",how="left")

# COMMAND ----------

# joining shoppers_df to Products
shoppers_df = shoppers_df.merge(products,on="ProductID",how="left")

# COMMAND ----------

# joining shoppers_df to Payments
shoppers_df = shoppers_df.merge(payments,on="OrderID",how="left")

# COMMAND ----------

# MAGIC %md
# MAGIC The tables are now joined and now saved as shoppers_df.  The new dataset will be inspected to check if there are no data quality issues that arose before feature engineering, analysis and answering the business questions.

# COMMAND ----------

# checking for the shape of the new dataset
shoppers_df.shape

# COMMAND ----------

# checking for summaries of the new dataset
shoppers_df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC After joining the datasets, 30 records had missing customer information, including Age, City, SignupDate, CustomerSegment and AgeGroup. Further investigation showed that these orders did not have matching CustomerIDs in the Customers dataset (they had a dummy CustomerID). The records were retained because their order, product and payment information was complete, but they will be excluded from customer demographic analysis.

# COMMAND ----------

# flagging the issue with SignupDate
shoppers_df["SignupDateIssue"] = (shoppers_df["OrderDate"] < shoppers_df["SignupDate"])
shoppers_df["SignupDateIssue"].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC A consistency check between customer SignupDate and OrderDate identified 13,898 orders where the order date occurred before the customer's recorded signup date. This is a large number and indicates a potential data quality issue with the customer signup dates. Due to the large number of affected records and the absence of information to determine the correct dates, the records were retained rather than removed or modified. The issue was flagged and will be considered when interpreting analyses involving customer signup dates or customer tenure.

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Feature engineering
# MAGIC In this section we are going to create new features/variables such as AgeGroups, Revenue and months and year extracts. This allows for detailed analysis to be made later on.

# COMMAND ----------

# MAGIC %md
# MAGIC **Creating Age Buckets**

# COMMAND ----------

#creating customer Age buckets
customers["AgeGroup"] = pd.cut(customers["Age"], bins=[18, 24, 34, 44, 54, 65],
    labels=["18-27", "28-37", "38-47", "48-57", "58-65"],
    include_lowest=True)

# COMMAND ----------

# checking to see if the AgeGroup buckets were created
customers['AgeGroup'].value_counts().sort_index()

# COMMAND ----------

# MAGIC %md
# MAGIC Due to a big gap between minimum and maximum Ages, age group buckets were created with the assumption that people of the same age have similar behavioural traits. Six age group buckets were created and confirmed. A large number of customers fall within 38-47, 28-37 and 58-65 age buckets.

# COMMAND ----------

# MAGIC %md
# MAGIC **Extracting time periods**

# COMMAND ----------

# extracting year from OrderDate and creating a new year column
shoppers_df["Year"] = shoppers_df["OrderDate"].dt.year

# COMMAND ----------

# extracting month from OrderDate and creating a new month column
shoppers_df["Month"] = shoppers_df["OrderDate"].dt.month

# COMMAND ----------

# creating month name column
shoppers_df["MonthName"] = shoppers_df["OrderDate"].dt.month_name()

# COMMAND ----------

# creating dayname column
shoppers_df["DayName"] = shoppers_df["OrderDate"].dt.day_name()

# COMMAND ----------

# MAGIC %md
# MAGIC Time periods were created using OrderDate column. These periods include year, month and monthname and an additional dayname column was created.

# COMMAND ----------

# MAGIC %md
# MAGIC **Calculating revenue**

# COMMAND ----------

# Calculating revenue after discount
shoppers_df["UnfilteredRevenue"] = (shoppers_df["Quantity"] *shoppers_df["UnitPrice"] *(1 - shoppers_df["Discount"])).round(2)

# COMMAND ----------

# checking to see the unique value counts in Status and PaymentStatus column
print(shoppers_df["Status"].value_counts())
print(shoppers_df["PaymentStatus"].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC A value count check on the Status and PaymentStatus column was checked to see the unique value counts before making a decision. It shows that the ordes that were not complete or not paid for are over 7 000, incorperating them might overstate our revenue so a decision is made to exclude them in the realised revenue.

# COMMAND ----------

# creating a new clean revenue column that has only completed and paid orders
shoppers_df["RealisedRevenue"] = shoppers_df["UnfilteredRevenue"].where((shoppers_df["Status"] == "Completed") & (shoppers_df["PaymentStatus"] == "Paid"),0)

# COMMAND ----------

# MAGIC %md
# MAGIC Revenue was calculated by multiplying Quantity by UnitPrice and adjusting for the Discount applied to each order (Quantity x UnitPrice x (1 - Discount). Year, Month and MonthName were extracted from OrderDate to support trend analysis over time. Cancelled, returned and unpaid orders were retained in the dataset for further analysis but were excluded from RealisedRevenue because they do not represent successfully completed and paid sales.

# COMMAND ----------

# MAGIC %md
# MAGIC # 7. Answering business questions
# MAGIC In this notebook, only the business question will be answered. The rest of the questions will be answered using google sheets visualisations and pivot tables.

# COMMAND ----------

# calculating shoppers' total unfiltered revenue
shoppers_df["UnfilteredRevenue"].sum()

# COMMAND ----------

# calculating shoppers filtered/clean revenue
shoppers_df["RealisedRevenue"].sum()

# COMMAND ----------

# calculating the diference between revenue before deductions and revenue after deductions
revenue_difference = (shoppers_df["UnfilteredRevenue"].sum()-shoppers_df["RealisedRevenue"].sum()).round(2)

display (revenue_difference)

# COMMAND ----------

# calculating percentage difference between revenue before deductions and revenue after deductions
revenue_difference = (((shoppers_df["UnfilteredRevenue"].sum()-shoppers_df["RealisedRevenue"].sum())/(shoppers_df["UnfilteredRevenue"].sum()))*100).round(2)

display(revenue_difference)

# COMMAND ----------

# MAGIC %md
# MAGIC A separate DataFrame was created to exclude cancelled, returned, and unpaid orders from the revenue calculation. The comparison shows a significant difference between the two results, with a revenue difference of R502,273.20, representing 14.44% of the total revenue when all orders are included. Based on this finding, cancelled, returned, and unpaid orders should not be counted as revenue, as they do not represent successfully completed sales from which the business earned revenue. I will therefore go back and add a new clean revenue column that will be used on all further revenue analysis.

# COMMAND ----------

# MAGIC %md
# MAGIC * ### Data cleaning extension
# MAGIC This section is an an extension of the data cleaning process because it was noticed after joining the tables that the place holder that is on CustomerID resulted i duplicates and changing it during the normal data presentation will not give a true picture. Object columns will be replaced by an Unknown

# COMMAND ----------

# replacising the CustomerID placeholder with an unknown so that it does not show as a null in the final output
shoppers_df.loc[shoppers_df["CustomerID"] == 999999, "City"] = "Unknown"
shoppers_df.loc[shoppers_df["CustomerID"] == 999999, "CustomerSegment"] = "Unknown"

# COMMAND ----------

shoppers_df[shoppers_df["City"].isna()][
    ["CustomerID", "Age", "City", "SignupDate", "CustomerSegment"]
].head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC CustomerID 999999 was identified as a placeholder customer ID. These records could not be matched to the customer dataset, resulting in missing Age, City, SignupDate and CustomerSegment information. The transactions were retained because their order and revenue information remained valid. City and CustomerSegment were classified as Unknown, while Age and SignupDate were left missing to avoid introducing artificial values.

# COMMAND ----------

# MAGIC %md
# MAGIC * ### Downloading data
# MAGIC The final dataset will be downloaded for analysis in google sheets.

# COMMAND ----------

# downloading the dataframe as a csv file 
display (shoppers_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion
# MAGIC This analysis explored the performance of the online store from 1 January 2024 to 30 June 2026. Four datasets covering customers, orders, products and payments were loaded and inspected separately. Data cleaning was performed to address missing values, incorrect data types and other data quality issues. Additional features, including customer age groups and other relevant categories, were created to support the analysis. The four datasets were then joined using CustomerID, ProductID and OrderID to create one combined dataset named shoppers_df. The dataset was transfered to google sheets for further analysis to understand customer behaviour, sales, products and payment patterns.