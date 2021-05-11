import io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from PyPDF2 import PdfFileReader, PdfFileWriter

from reportlab.lib.pagesizes import letter


def pdf_prossessing(RECAP_Sprint_best, RECAP_Sprint_last, INFOS):
    from reportlab.pdfgen import canvas
    fatigue_index = round(((RECAP_Sprint_last["Temps_sprint"] - RECAP_Sprint_best["Temps_sprint"]) / RECAP_Sprint_last["Temps_sprint"])*100, 1)

    from math import pi
# graph radar
    # Set data
    df = pd.DataFrame({
        'Sprint': ['6', 'Best'],
        'Vmoy start': [RECAP_Sprint_last["Vmoy_start_D&G"], RECAP_Sprint_best["Vmoy_start_D&G"]],
        'Vitesse max': [RECAP_Sprint_last["Vitesse_max"], RECAP_Sprint_best["Vitesse_max"]],
        'Vmoy stabilisee': [RECAP_Sprint_last["Vmoy_stab_D&G"], RECAP_Sprint_best["Vmoy_stab_D&G"]],
        'Vitesse moyenne': [RECAP_Sprint_last["Vitesse_moyenne"], RECAP_Sprint_best["Vitesse_moyenne"]],
        'Distance pour Vmax': [RECAP_Sprint_last["Distance_Vmax"], RECAP_Sprint_best["Distance_Vmax"]],
        'Temps sprint': [RECAP_Sprint_last["Temps_sprint"], RECAP_Sprint_best["Temps_sprint"]]
    })

    categories = list(df)[1:]
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    ax = plt.subplot(111, polar=True)

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories)

    ax.set_rlabel_position(0)
    plt.yticks([10, 15, 20], ["10", "15", "20"], color="grey", size=7)
    plt.ylim(0, 20)

    # Ind1
    values = df.loc[0].drop('Sprint').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="Sprint 6")
    ax.fill(angles, values, 'r', alpha=0.1)

    # Ind2
    values = df.loc[1].drop('Sprint').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="meilleur sprint")
    ax.fill(angles, values, 'b', alpha=0.1)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # Show the graph
    # plt.show()
    plt.savefig("plot.png")
    plt.close()

# Graph bar
    barWidth = 0.3


    bars1 = [RECAP_Sprint_best["Acceleration_start"], RECAP_Sprint_best["Acceleration_moyenne"]]

    bars2 = [RECAP_Sprint_last["Acceleration_start"], RECAP_Sprint_last["Acceleration_moyenne"]]

    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]

    plt.bar(r1, bars1, width=barWidth, color='blue', edgecolor='black', capsize=7, label='Meilleur sprint')

    plt.bar(r2, bars2, width=barWidth, color='red', edgecolor='black', capsize=7, label='Sprint 6')

    plt.xticks([r + barWidth for r in range(len(bars1))], ['Acceleration au demarrage', 'Acceleration moyenne'])
    plt.ylabel('height')
    plt.legend()

    # plt.show()
    plt.savefig("plotbar.png")

# graph vitesse sprint


# PDF page 1
    packet = io.BytesIO()
    canvas = canvas.Canvas(packet, pagesize=letter)

    canvas.drawImage('plot.png', 52, 270, width=280, height=230, preserveAspectRatio=False)
    canvas.drawImage('plotbar.png', 290, 270, width=250, height=240, preserveAspectRatio=False)

    canvas.setLineWidth(.3)
    canvas.setFont('Helvetica', 11)
    canvas.drawString(340, 585, INFOS["angle_carossage"])
    canvas.drawString(340, 570, INFOS["taille_roues"])
    canvas.drawString(150, 600, INFOS["date"])
    canvas.drawString(150, 197, str(RECAP_Sprint_best["Vmoy_start_D&G"]))
    canvas.drawString(150, 182, str(RECAP_Sprint_best["Vmoy_stab_D&G"]))
    canvas.drawString(150, 167, str(RECAP_Sprint_best["Vitesse_moyenne"]))
    canvas.drawString(150, 152, str(RECAP_Sprint_best["Vitesse_max"]))
    canvas.drawString(150, 137, str(RECAP_Sprint_best["Distance_Vmax"]))
    canvas.drawString(150, 122, str(RECAP_Sprint_best["Acceleration_start"]))
    canvas.drawString(150, 107, str(RECAP_Sprint_best["Acceleration_stab"]))
    canvas.drawString(150, 92, str(RECAP_Sprint_best["Acceleration_moyenne"]))
    canvas.drawString(150, 74, str(RECAP_Sprint_best["Temps_sprint"]))

    canvas.drawString(190, 197, str(RECAP_Sprint_last["Vmoy_start_D&G"]))
    canvas.drawString(190, 182, str(RECAP_Sprint_last["Vmoy_stab_D&G"]))
    canvas.drawString(190, 167, str(RECAP_Sprint_last["Vitesse_moyenne"]))
    canvas.drawString(190, 152, str(RECAP_Sprint_last["Vitesse_max"]))
    canvas.drawString(190, 137, str(RECAP_Sprint_last["Distance_Vmax"]))
    canvas.drawString(190, 122, str(RECAP_Sprint_last["Acceleration_start"]))
    canvas.drawString(190, 107, str(RECAP_Sprint_last["Acceleration_stab"]))
    canvas.drawString(190, 92, str(RECAP_Sprint_last["Acceleration_moyenne"]))
    canvas.drawString(190, 74, str(RECAP_Sprint_last["Temps_sprint"]))
    canvas.drawString(170, 60, (str(fatigue_index) + " %"))
    canvas.save()
    packet.seek(0)



    new_pdf = PdfFileReader(packet)

    pdf_path = 'Model_Rapport_sprints_repetes.pdf'
    pdf = PdfFileReader(str(pdf_path))
    output = PdfFileWriter()
    page = pdf.getPage(0)
    page2 = pdf.getPage(1)
    page.mergePage(new_pdf.getPage(0))

    output.addPage(page)

    # PDF page 2
    from reportlab.pdfgen import canvas
    packet2 = io.BytesIO()
    c = canvas.Canvas(packet2, pagesize=letter)
    c.drawImage('all_sprint.png', 52, 450, width=500, height=300, preserveAspectRatio=False)
    c.save()
    packet2.seek(0)
    new_pdf2 = PdfFileReader(packet2)

    page2.mergePage(new_pdf2.getPage(0))


    output.addPage(page2)
    outputStream = open("Rapport_sprint_repetes" + INFOS["nom_test"] + ".pdf", "wb")
    output.write(outputStream)
    outputStream.close()
    return outputStream

