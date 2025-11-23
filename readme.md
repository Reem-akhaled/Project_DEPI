## 🇮🇳 Indian Banks Customer Reviews Analysis

An advanced Business Intelligence (BI) project analyzing 3,000 customer reviews for major Indian banks to benchmark performance, identify service quality gaps, and track temporal trends. The final output is an interactive Power BI dashboard designed for executive decision-making.

### 🌟 Project Goals

The core objective of this project was to transform raw, unstructured bank review data into clear, actionable business intelligence using a comprehensive data model and visualization techniques.

* **Bank Comparison:** Compare the performance of 12 major banks based on key indicators like the Composite Score and service quality.
* **Thematic Insights:** Analyze customer feedback using AI-driven text mining to identify common themes, recurring issues, and positive highlights.
* **Time-Based Monitoring:** Track how ratings, opinions, and review activity evolve over time to detect performance patterns and seasonality.

### 💡 Key Strategic Takeaways

The analysis provided clear, data-driven conclusions for service improvement and strategic focus:

* **Quality Leader Identified:** **Citibank** sets the service benchmark, achieving the highest Average Review Score (4.64) and a leading Composite Score (60.0%), with 100% of its reviews categorized as Excellent Service.
* **Root Cause of Negative Sentiment:** The thematic analysis points to two critical pain points: **unhelpful staff** and **very slow service**. This suggests operational efficiency and personnel training are immediate priorities for low-performing banks.
* **Service Recovery Required:** **Bank of Baroda** and **PNB** show the lowest average ratings and a high proportion of "Bad Service" reviews, requiring immediate intervention.
* **Engagement Volatility:** Negative Reviews, while lower in volume (29.9% of total), drive high and volatile audience engagement, peaking at **44K likes in 2023**. This necessitates a robust protocol for rapid response to high-impact negative feedback.
* **Seasonal Peaks:** Review volume exhibits a clear annual cycle, peaking in both **January** and **December**, indicating critical periods for service readiness and resource allocation.

### 🛠️ Technical Methodology

| Component | Detail |
| :--- | :--- |
| **Dataset Size** | 3,000 customer reviews across 12 banks and 325 cities. |
| **Data Modeling** | Implemented a **Hybrid Schema (Starflake)** design for performance and analytical depth. |
| **Text Analysis** | An AI-generated Python script was executed for text mining, isolating top keywords, and generating a Word Cloud. |
| **Key Metric** | **Bank Composite Score:** A normalized performance score combining **Rating (60%), Engagement (20%), and Share of Voice (20%)**. |
| **Data Enrichment** | Raw ratings were categorized into **Bad, Good, and Excellent Service** quality levels for granular analysis. |

### 🖼️ Dashboard Components

The interactive dashboard is split into four distinct analysis pages:

1.  **Overview:** Provides a high-level snapshot with KPIs, Engagement Summary, and a **Ratings vs. Reviews Scatter Plot**.
2.  **Bank Analysis:** Ranks all banks by the **Composite Score** and provides deep drill-down on individual bank performance, including its specific feedback breakdown.
3.  **Review Insights:** Integrates text analysis output, showcasing the **Word Cloud**, Most Frequent Words per sentiment, and the Most Helpful Review.
4.  **Time-Based Trends:** Features dynamic trend visuals to track review volume and engagement MoM/YoY across All Reviews, Positive, Neutral, and Negative segments.

***


To view the interactive dashboard:

1.  **Download:** Download the `Banks Reviews.pbix` file.
2.  **Software:** Ensure you have **Power BI Desktop** installed.
3.  **Run:** Open the `.pbix` file to interact with the full data model and visualizations.

## 🖼️ Dashboard Preview

![Dashboard Screenshot](Dashboard_Overview.PNG)