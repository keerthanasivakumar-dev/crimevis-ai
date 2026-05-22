from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
import datetime
import os

now = datetime.datetime.now()
filename = "CRIMEVIS_Report.pdf"

incidents_folder = "incidents"
incidents = []

if os.path.exists(incidents_folder):
    incidents = sorted([f for f in os.listdir(incidents_folder) if f.endswith('.jpg')])

total = len(incidents)

doc = SimpleDocTemplate(filename, pagesize=A4,
    leftMargin=15*mm, rightMargin=15*mm,
    topMargin=15*mm, bottomMargin=15*mm)

styles = getSampleStyleSheet()
story = []

title_style = ParagraphStyle('T', parent=styles['Normal'],
    fontSize=20, textColor=colors.red,
    fontName='Helvetica-Bold', alignment=1, spaceAfter=5)

sub_style = ParagraphStyle('S', parent=styles['Normal'],
    fontSize=10, textColor=colors.grey,
    alignment=1, spaceAfter=20)

body_style = ParagraphStyle('B', parent=styles['Normal'],
    fontSize=10, textColor=colors.black, spaceAfter=5)

story.append(Paragraph("CRIMEVIS AI", title_style))
story.append(Paragraph("Automated Incident Report", sub_style))
story.append(Paragraph(f"Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}", body_style))
story.append(Paragraph(f"Total Incidents: {total}", body_style))
story.append(Spacer(1, 10))

data = [
    ["Report Date", now.strftime('%Y-%m-%d')],
    ["Report Time", now.strftime('%H:%M:%S')],
    ["Total Incidents", str(total)],
    ["System", "CRIMEVIS AI v1.0"],
    ["Detection Engine", "YOLOv8 + OpenCV"],
]

t = Table(data, colWidths=[60*mm, 110*mm])
t.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (0,-1), colors.red),
    ('TEXTCOLOR', (0,0), (0,-1), colors.white),
    ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
    ('FONTSIZE', (0,0), (-1,-1), 9),
    ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ('PADDING', (0,0), (-1,-1), 6),
    ('BACKGROUND', (1,0), (1,-1), colors.HexColor('#F5F5F5')),
]))
story.append(t)
story.append(Spacer(1, 15))

doc.build(story)
print(f"PDF REPORT GENERATED: {filename}")
