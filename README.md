<img width="1920" height="760" alt="BRIGHTLEARN COFFEE SHOP SALES ANALYSIS REPORT" src="https://github.com/user-attachments/assets/a983adaa-6ce7-436c-bc91-0cc76026a914" />


## Project Overview

This project analyses an online store's performance from January 2024 to June 2026. The analysis was conducted for the Head of Operations to provide a clear understanding of the store's financial, product, customer, and operational performance. Four datasets were used for the analysis, namely, Customers, Orders, Payments, and Products. The datasets were inspected and cleaned separately before being joined to create a final dataset for analysis.

## Project Objectives

The main objectives of the project were to:

- Measure the store's revenue, valid orders, and Average Order Value (AOV).
- Analyse monthly revenue trends and identify possible seasonal patterns.
- Evaluate product and category performance based on revenue and units sold.
- Identify the most valuable cities and customer segments.
- Assess revenue leakage through cancelled orders, returned orders, and failed payments.
- Determine whether higher discounts increase basket size or mainly reduce revenue.
- Translate the analysis into clear business insights and practical recommendations.

## Business Questions

The analysis aimed to answer the following questions:

1. How much revenue did the shop make, from how many orders, and what was the Average Order Value (AOV)?
2. Is revenue growing, shrinking, or remaining flat month by month? Are there any seasonal peaks?
3. Which products and categories generate the most revenue, and which sell the most units?
4. Which cities and customer segments (New, Regular, VIP) are the most valuable?
5. What proportion of orders are cancelled or returned, and how frequently do payments fail?
6. Do larger discounts result in bigger orders, or do they mainly reduce revenue?

## Dataset Overview

The project used four datasets:

| Dataset | Description |
|---|---|
| **Customers** | Customer age, city, signup date and customer segment |
| **Orders** | Order date, products, quantities, discounts, payment method and order status |
| **Payments** | Payment attempts and payment status |
| **Products** | Product names, categories and unit prices |

### Dataset Relationships

The datasets were joined using:

- `CustomerID` – Orders to Customers
- `ProductID` – Orders to Products
- `OrderID` – Orders to Payments

## Project Workflow

### 1. Project Planning

The project started with understanding the business problem and developing a data analysis plan in Miro. The planning stage outlined the business questions, datasets, data preparation requirements, KPIs, analysis approach and expected deliverables.

### 2. Data Inspection and Cleaning

The four datasets were loaded into Databricks and converted into Pandas DataFrames. Each dataset was inspected separately before joining.

The data preparation process included:

- Checking dataset dimensions and structure
- Inspecting data types
- Identifying missing values
- Checking duplicates
- Investigating inconsistent values
- Converting date columns to the correct format
- Validating IDs and relationships between datasets

### 3. Data Integration

After cleaning, the datasets were joined to create a single analysis-ready dataset.

The main relationships were:

`Orders → Customers` using `CustomerID`

`Orders → Products` using `ProductID`

`Orders → Payments` using `OrderID`

Checks were performed after joining to ensure that the relationships did not unexpectedly change the dataset.

### 4. Feature Engineering

Additional variables were created to support the analysis, including Year, Month and Revenue.

Revenue was calculated as:

`Revenue = Quantity × UnitPrice × (1 - Discount)`

For realised revenue analysis, cancelled, returned, failed, refunded and other non-paid transactions were excluded where applicable.

### 5. Data Analysis and Visualisation

The processed data was analysed using pivot tables and charts in Google Sheets.

The analysis focused on:

- Revenue and order performance
- Monthly revenue trends
- Product and category performance
- Customer segments
- Geographical performance
- Cancelled and returned orders
- Payment failures
- Discount effectiveness

## Key Performance Indicators

The main KPIs and measures included:

- Realised Revenue
- Valid Orders
- Average Order Value (AOV)
- Monthly Revenue
- Units Sold
- Revenue by Product and Category
- Revenue by City
- Revenue by Customer Segment
- Cancellation Rate
- Return Rate
- Payment Failure Rate
- Average Quantity by Discount Level
- Average Order Revenue by Discount Level

## Tools

- **Miro** – Data analysis and project planning
- **Databricks / Python / Pandas** – Data cleaning, processing, joining, feature engineering and analysis
- **Google Sheets** – Pivot tables and data visualisation
- **Canva** – Final presentation
- **GitHub** – Project documentation and portfolio

## Project Deliverables

The project produced:

- Data analysis plan
- Cleaned and processed dataset
- Databricks analysis notebook
- Pivot tables
- Data visualisations
- Business insights and recommendations
- Final presentation

## Conclusion

This project demonstrates an end-to-end data analysis workflow, from understanding the business problem and planning the analysis to cleaning and integrating multiple datasets, analysing business performance, creating visualisations and communicating actionable insights. The project also demonstrates the ability to translate technical data analysis into clear information that can support operational decision-making.
