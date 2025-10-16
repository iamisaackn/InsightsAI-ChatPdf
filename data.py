import pandas as pd
import random
from faker import Faker
from datetime import timedelta
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

fake = Faker()

# --- Config ---
num_records = 10
products = ["Laptop", "Phone", "Headphones", "Printer", "Monitor"]
regions = ["Nairobi", "Mombasa", "Kisumu"]
segments = ["Retail", "Wholesale", "Online"]

# --- 1. Sales Logs ---
sales_data = []
for _ in range(num_records):
    date = fake.date_between(start_date="-3M", end_date="today")
    product = random.choice(products)
    units_sold = random.randint(0, 20)
    unit_price = random.randint(50, 200) * 10
    segment = random.choice(segments)
    region = random.choice(regions)
    
    # seasonal spikes
    if date.day > 25 and random.random() < 0.3:
        units_sold *= 3

    sales_data.append([date, product, units_sold, unit_price, segment, region])

sales_df = pd.DataFrame(sales_data, columns=["Date", "ProductID", "UnitsSold", "UnitPrice", "CustomerSegment", "Region"])

# --- 2. Stock Reports ---
stock_data = []
for _ in range(num_records):
    date = fake.date_between(start_date="-3M", end_date="today")
    product = random.choice(products)
    opening = random.randint(0, 50)
    sold = random.randint(0, opening) if opening > 0 else 0
    closing = opening - sold
    reorder_level = random.randint(10, 30) if closing == 0 and random.random() < 0.5 else random.randint(5, 15)

    stock_data.append([date, product, opening, sold, closing, reorder_level])

stock_df = pd.DataFrame(stock_data, columns=["Date", "ProductID", "OpeningStock", "UnitsSold", "ClosingStock", "ReorderLevel"])

# --- 3. Cash Flow / Invoices ---
invoice_data = []
for _ in range(num_records):
    date = fake.date_between(start_date="-3M", end_date="today")
    invoice_id = fake.uuid4()[:8]
    customer_id = fake.uuid4()[:6]
    amount = random.randint(100, 2000)
    
    if random.random() < 0.7:
        status = "Paid"
        payment_date = date + timedelta(days=random.randint(0, 10))
        if random.random() < 0.3:  # late payment
            payment_date = date + timedelta(days=random.randint(15, 30))
    else:
        status = "Unpaid"
        payment_date = None

    invoice_data.append([date, invoice_id, customer_id, amount, status, payment_date])

invoice_df = pd.DataFrame(invoice_data, columns=["Date", "InvoiceID", "CustomerID", "Amount", "Status", "PaymentDate"])

# --- Convert DataFrames to descriptive sentences ---
def df_to_sentences(df, template_func):
    return [template_func(row) for _, row in df.iterrows()]

sales_template = lambda r: f"On {r.Date}, {r.UnitsSold} units of {r.ProductID} were sold at {r.UnitPrice} each in the {r.CustomerSegment} segment at {r.Region}."
stock_template = lambda r: f"On {r.Date}, {r.ProductID} had opening stock {r.OpeningStock}, sold {r.UnitsSold}, leaving {r.ClosingStock} units, with reorder level {r.ReorderLevel}."
invoice_template = lambda r: f"Invoice {r.InvoiceID} for customer {r.CustomerID} dated {r.Date} amounted to {r.Amount} and is {r.Status}. Payment date: {r.PaymentDate}."

sales_sentences = df_to_sentences(sales_df, sales_template)
stock_sentences = df_to_sentences(stock_df, stock_template)
invoice_sentences = df_to_sentences(invoice_df, invoice_template)

all_sentences = sales_sentences + stock_sentences + invoice_sentences

# --- Save descriptive sentences to text file ---
with open("business_data.txt", "w") as f:
    for sentence in all_sentences:
        f.write(sentence + "\n")

print("Saved business_data.txt with descriptive sentences for embeddings.")

# --- Helper: Save DataFrame to PDF ---
def df_to_pdf(df, filename, title):
    doc = SimpleDocTemplate(filename, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    
    elements.append(Paragraph(title, styles['Title']))
    elements.append(Spacer(1, 12))
    
    data = [df.columns.tolist()] + df.values.tolist()
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#4F81BD")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    
    elements.append(table)
    doc.build(elements)
    print(f"Saved: {filename}")

# --- Export PDFs ---
df_to_pdf(sales_df, "sales_logs.pdf", "Sales Logs")
df_to_pdf(stock_df, "stock_reports.pdf", "Stock Reports")
df_to_pdf(invoice_df, "invoices.pdf", "Cash Flow / Invoices")
