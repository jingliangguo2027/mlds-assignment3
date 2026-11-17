"""Build the PDF from the notebook's exported metrics and figures."""
import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak,
)

ROOT = Path(__file__).resolve().parent
D = json.loads((ROOT / 'experiment_results.json').read_text())
OUT = ROOT / 'Assignment3_Report.pdf'
BLUE = colors.HexColor('#173e62')
GRAY = colors.HexColor('#53616d')
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyReport', fontName='Helvetica', fontSize=10.1, leading=14, spaceAfter=8))
styles.add(ParagraphStyle(name='TitleReport', fontName='Helvetica-Bold', fontSize=23, leading=28, textColor=BLUE, spaceAfter=10))
styles.add(ParagraphStyle(name='SectionReport', fontName='Helvetica-Bold', fontSize=17, leading=22, textColor=BLUE, spaceAfter=12))
styles.add(ParagraphStyle(name='SubReport', fontName='Helvetica-Bold', fontSize=11.5, leading=15, textColor=BLUE, spaceBefore=6, spaceAfter=6))
styles.add(ParagraphStyle(name='CaptionReport', fontName='Helvetica', fontSize=8.5, leading=11, textColor=GRAY, spaceAfter=10))
styles.add(ParagraphStyle(name='CellReport', fontName='Helvetica', fontSize=9, leading=12))
story = []


def p(text, style='BodyReport'):
    story.append(Paragraph(text, styles[style]))


def title(text):
    p(text, 'SectionReport')


def sub(text):
    p(text, 'SubReport')


def figure(name, width=6.65):
    image = Image(str(ROOT / 'figures' / name))
    image.drawHeight *= width * inch / image.drawWidth
    image.drawWidth = width * inch
    story.append(image)
    story.append(Spacer(1, 6))


def caption(text):
    p(text, 'CaptionReport')


def table(rows, widths):
    content = [[Paragraph(escape(str(x)), styles['CellReport']) for x in row] for row in rows]
    t = Table(content, colWidths=[x * inch for x in widths], repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e9f0f6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), BLUE),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LINEBELOW', (0, 0), (-1, 0), .7, BLUE),
        ('LINEBELOW', (0, 1), (-1, -1), .3, colors.HexColor('#d6dde3')),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))


def page():
    story.append(PageBreak())


def percent(value):
    return f'{value:.1%}'


def link(label, url):
    return f'<link href="{url}" color="#173e62">{label}</link>'


data, seq, ray, val = D['data'], D['sequential'], D['ray'], D['validation']
noise = D['noise']
counts = np.array(data['class_counts'])
coverage = np.array(data['coverage'])
sizes = np.array(data['sizes'])

p('Federated Learning<br/>and Differential Privacy', 'TitleReport')
p('Assignment 3 | Jingliang Guo | November 17, 2026', 'SubReport')
p('GitHub username <b>jingliangguo2027</b><br/>Public repository <b>mlds-assignment3</b><br/>' + link('github.com/jingliangguo2027/mlds-assignment3', 'https://github.com/jingliangguo2027/mlds-assignment3'))
sub('Data and model')
p(f'The supplied archive contains {data["clients"]} clients with {data["total"]:,} local images and {data["test"]:,} held-out test images. Each 28 x 28 grayscale image belongs to one of 62 classes. I keep the supplied pixel values and flatten each image into 784 inputs. The model is Linear(784, 128), ReLU, and Linear(128, 62), trained with cross-entropy, SGD, learning rate 0.01, and batch size 32.')
p(f'Each client has an independent shuffled 80/20 split with seed 42, giving {data["train"]:,} training and {data["validation"]:,} validation samples. Local client sizes range from {sizes.min()} to {sizes.max()} with median {np.median(sizes):.0f}. The most common overall class is {counts.argmax()} ({counts.max() / counts.sum():.1%} of images).')
figure('class_distribution.png', 6.1)
caption('Figure 1. Overall and five selected client class distributions. The common percentage axis exposes client heterogeneity. Class identifiers follow the provided dataset.')
p(f'Clients contain {coverage.min()} to {coverage.max()} observed classes (median {np.median(coverage):.0f}). The model retains all 62 outputs even when a client has missing classes. Only client model parameters and training sample counts enter server aggregation.', 'BodyReport')

page()
title('Part 1 | Sequential FedAvg')
p('At each communication round, I sample clients without replacement. Every selected client starts from the same global parameters and trains on its own split. The server forms the sample-weighted mean of the resulting parameters. For client k with n_k local training examples, its aggregation weight is n_k divided by the sum of the selected clients\' training counts.')
figure('sequential_curves.png')
caption('Figure 2. Sample-weighted training and validation loss and accuracy for 20 rounds, C=10% and E=3. Evaluation uses the updated global model on all 100 local splits.')
sub('Hyperparameter comparison')
rows = [['Clients per round', 'Local epochs E', 'Train accuracy', 'Validation accuracy']]
rows += [[f'{g["clients"]} ({g["clients"]}%)', g['epochs'], percent(g['train']), percent(g['validation'])] for g in seq['grid']]
table(rows, [1.9, 1.3, 1.6, 1.85])
p(f'Each comparison uses 50 rounds and the same initialization and client-permutation seed. The best validation setting is C=10% and E=3. Its held-out test accuracy is <b>{seq["test"]:.1%}</b>, with test loss {seq["test_loss"]:.3f}. Test images take no part in model updates or the selection of this setting.')
p('Increasing E improves validation accuracy at both tested participation rates. Increasing C also improves accuracy in this grid. These changes increase the local work per communication round. The comparison measures accuracy at a fixed communication budget. The low absolute accuracy and falling loss motivate a wider learning-rate and training-budget search in future work.')

page()
title('Part 1 | Parallel clients with Ray Core')
p('Each round samples four clients. I create one Actor per selected client, reserve one CPU per Actor, and run each local model on CPU with one PyTorch compute thread. All methods are submitted before their results are collected. Actors return a model state and a training sample count. The shared aggregation function applies the same weighted average as sequential FedAvg.')
p(f'The local Ray cluster exposes {D["environment"]["ray_cpus"]:.0f} CPUs. The implementation processes selected clients in batches bounded by the cluster CPU count, with Actor cleanup in a finally block. This releases resources before later batches. Training uses E=3 for 20 rounds. The global evaluation device is {D["environment"]["device"]}.')
figure('ray_curves.png')
caption('Figure 3. Ray FedAvg with four clients per round. Final validation accuracy is ' + percent(ray['history']['val'][-1][1]) + '; test accuracy is ' + percent(ray['test']) + '.')
sub('Matched execution check')
p(f'I held the initial CPU model, client identities, per-client seeds, local epochs, and round count fixed between serial execution and Ray. The maximum parameter difference over three rounds was <b>{max(val["matched_errors"]):.2e}</b>. This directly checks that changing execution scheduling preserves the update under matched conditions.')
table([['Three matched rounds', 'Update time'], ['Serial CPU', f'{val["serial_seconds"]:.3f} seconds'], ['Ray Actors', f'{val["ray_seconds"]:.3f} seconds']], [4.5, 2.15])
p('Timing includes Actor construction and cleanup. These tiny local jobs make process and serialization overhead substantial. The earlier 10-client, 50-round model and the four-client, 20-round model have different training budgets, so their accuracy gap combines these configuration changes. [2]')

page()
title('Part 2 | Laplace input perturbation')
p('For every training pixel I independently sample zero-centered Laplace noise with scale b and add it to the supplied image. For each b, the perturbation is sampled once per training image and reused over all 50 rounds. Values retain their full perturbed range. Validation and test images stay clean. All runs use C=10%, E=3, the same split, initialization seed 42, and client-permutation seed 42.')
p('Training accuracy is measured on the original training images to keep evaluation inputs comparable across b. The local optimizer sees the perturbed images. The following figures show both the communication trajectory and the final training/test comparison.')
figure('noise_round_curves.png')
caption('Figure 4. Global accuracy on clean training and validation splits for each perturbation scale. Each curve uses 50 communication rounds.')
figure('noise_final_accuracy.png', 4.65)
caption('Figure 5. Final clean-training accuracy and held-out test accuracy versus Laplace scale. Accuracy is non-monotonic within this tested range.')

page()
title('Part 2 | Results and scale choice')
rows = [['Noise scale b', 'Train accuracy', 'Validation accuracy', 'Test accuracy']]
rows += [[b, percent(v['train_curve'][-1]), percent(v['validation_curve'][-1]), percent(v['test'])] for b, v in noise.items()]
table(rows, [1.5, 1.6, 1.8, 1.75])
p('In the seed-42 comparison, accuracy dips slightly at b=0.1, rises at b=0.3, and falls at b=0.6. I choose <b>b=0.3</b> for this comparison because it gives the highest validation accuracy among the perturbed settings. It also gives the highest observed test accuracy. The trend can reflect a regularization effect, which the repeated-seed experiment examines separately.')
figure('noise_seed_robustness.png', 4.4)
caption('Figure 6. Mean validation accuracy and sample standard deviation over three initialization/client-sampling seeds (42, 43, 44). The client split remains fixed. Error bars describe seed variability.')
rows = [['Noise scale b', 'Mean validation', 'Sample SD']]
for b, values in noise.items():
    scores = values['seed_validation']
    rows.append([b, percent(np.mean(scores)), percent(np.std(scores, ddof=1))])
table(rows, [2.15, 2.25, 2.25])
p('Across three seeds, b=0.3 has the highest mean validation accuracy. Overlapping error bars and one fixed split limit confidence in the scale choice.')

page()
title('Per-client validation')
sub('Majority-class baseline')
figure('client_baseline.png', 3.0)
caption('Figure 7. Each point is one client. The horizontal axis uses the client\'s training majority class to predict its validation labels. The vertical axis uses the selected global model. The diagonal marks equal accuracy.')
p(f'FedAvg exceeds this locally defined baseline for <b>{seq["baseline_wins"]}/100 clients</b>. The baseline uses each client\'s own class imbalance and never selects its class using validation labels. This checks whether the global model learns beyond a simple distribution-based prediction. The two remaining clients should be inspected separately during a broader tuning study.')
sub('Unequal impact of noise')
figure('noise_client_effect.png', 4.4)
caption('Figure 8. Per-client validation accuracy change at b=0.6 relative to b=0, plotted against local training sample count. This is the Part 2 non-standard plot.')
p(f'The median client change is {val["median_client_change"]:+.2f} percentage points. The scatter exposes clients that lose accuracy even when aggregate accuracy improves. Figures 7 and 8 complement the sample-weighted aggregate, where larger clients contribute more.')

page()
title('Validation experiments and predictions')
p('Run Section 6 of the notebook after Sections 1-5. Each experiment states a limit or counterfactual and prints its numerical outcome. Assertions check the corresponding prediction. Section 7 exports the evidence to experiment_results.json.')
rows = [
 ['Experiment', 'Prediction and observed evidence'],
 ['Unequal client aggregation', f'Two clients have 275 and 297 training images. The explicit weighted mean matches FedAvg with maximum parameter error {seq["aggregation_error"]:.2e}. This catches accidental equal weighting.'],
 ['Single-client limit', f'With one selected client, FedAvg equals its local update. Maximum parameter error is {val["single_client_error"]:.2e}.'],
 ['Zero learning rate', f'With lr=0, the global parameters stay fixed through local training and aggregation. Maximum change is {val["zero_lr_error"]:.2e}.'],
 ['Controlled 16-image fit', f'The same network fits a small real-data task. Loss changes from {val["tiny_before"][0]:.3f} to {val["tiny_after"][0]:.3f}; training accuracy changes from {percent(val["tiny_before"][1])} to {percent(val["tiny_after"][1])} after 2000 local epochs at lr=0.01 on CPU.'],
 ['Serial/Ray equivalence', f'Under matched CPU seeds and data, three rounds have maximum parameter errors ' + ', '.join(f'{e:.2e}' for e in val['matched_errors']) + '.'],
 ['Zero-noise recovery', f'b=0 preserves every training pixel exactly. Its final validation accuracy matches the C=10%, E=3 sequential run within 1e-6.'],
 ['Laplace moment check', f'For b=0.3, the signed sample mean is {val["noise_mean"]:.4f} and the mean absolute perturbation is {val["noise_abs_mean"]:.4f}. The expectations are 0 and 0.3. The tolerance is 2% of b.'],
 ['Repeated perturbation runs', 'For each of four scales, run 50 rounds with seeds 42, 43, and 44. Figure 6 reports mean and sample SD.'],
 ['High-noise stress', f'At b=3, validation accuracy is {percent(val["stress_validation"])} and {percent(val["stress_outside_fraction"])} of perturbed pixels fall outside [0,1]. Compare with the matched b=0 control.'],
]
table(rows, [1.65, 5.0])
p('The small-data fit isolates optimization capacity. The three controlled parameter comparisons isolate local-update, aggregation, and parallel-execution behavior. The noise moment and stress experiments test the mechanism and model sensitivity across substantially different input regimes.')

page()
title('Edge cases and handling')
sub('Part 1')
p(f'<b>Imbalanced clients and missing classes.</b> Sizes range from {sizes.min()} to {sizes.max()} images and class coverage from {coverage.min()} to {coverage.max()}. Aggregation uses training sample counts and retains a fixed 62-way classifier. Figure 1 describes heterogeneity, Figure 7 measures client-level utility, and the unequal-client calculation checks the weighting formula.')
p(f'<b>Small local datasets and split leakage.</b> Every client has at least two samples in this data. The split code explicitly requires that minimum, floors the 80% training size, and assigns the remainder to validation. It stores source indices and confirms zero overlap for all {data["clients"]} clients. Their train/validation sizes add back to the local total. Training and validation always remain within their original user.')
p('<b>Limited CPU resources and Actor cleanup.</b> Actors reserve one CPU and are created only for selected clients. When fewer than four CPUs are available, batches release their Actors before the next batch begins. A finally block also cleans up after failure. A failed round raises before the server loads a new aggregate, preserving the previous global model.')
p('<b>CPU and accelerator tensors.</b> Training and evaluation query the model parameter device before moving each input. Actor inputs are copied into writable tensors. This resolves the CPU/MPS mismatch and the warning caused by Ray\'s read-only NumPy arrays. Matched serial/Ray results check the resulting computation.')
sub('Part 2')
p('<b>Zero or invalid noise scales.</b> b=0 preserves images exactly and reproduces the matched baseline. Negative or non-finite scales raise ValueError. The reported accuracy curves include the zero-noise control.')
p(f'<b>Out-of-range pixels and strong perturbation.</b> Adding Laplace noise naturally produces values outside [0,1]. The implementation retains them to follow additive perturbation. At b=3, {val["stress_outside_fraction"]:.1%} of pixels exceed the original range and validation accuracy is {val["stress_validation"]:.1%}. The moment check confirms the scale in the original pixel units.')
p('<b>Unequal user impact and stochastic results.</b> Figure 8 reports individual client changes, while Figure 6 repeats the experiment across seeds. The fixed split makes comparisons paired. A broader study would vary the split and extend the seed count. These tests supply empirical uncertainty for the observed utility trend.')
sub('Interpretation of the privacy experiment')
p('This implementation follows the assignment\'s additive Laplace input mechanism. Labels remain available to local training, and the simulation process can access its data archive. A formal full-record or client-level privacy statement would require an explicit threat model and calibrated sensitivity for all protected information and released outputs. [3]')

page()
title('Reproduction and further work')
sub('Run the code')
p('Clone the public repository and obtain Assignment3-data.zip from Canvas. Put the ZIP next to reports/hw3.ipynb, then open a notebook kernel with the dependencies below. Run all notebook cells in order with reports/ as the working directory. Section 4 runs Ray, Section 5 runs the main noise experiment, and Section 6 runs the additional validations. Section 7 exports the numerical evidence.')
env = D['environment']
p(f'The verified environment uses Python {env["python"]}, PyTorch {env["torch"]}, NumPy {env["numpy"]}, and Ray {env["ray"]}. The global training device is {env["device"]}. Ray uses CPU Actors. Install the Python dependencies with <font name="Courier">python -m pip install -r requirements.txt</font> from the repository root. Select that same Python environment as the notebook kernel.')
p('The notebook writes figures/ and experiment_results.json beside itself. Its fixed seeds make the comparisons reproducible on the same device. Across accelerator types, small floating-point differences can change individual predictions. To rebuild the PDF after running the notebook, execute <font name="Courier">python reports/build_report.py</font> from the repository root.')
sub('Three additional months')
p('<b>Month 1.</b> Run a wider learning-rate, local-epoch, and communication-budget search using validation results. Repeat across at least ten training seeds and several independent per-client splits. Compare equal communication budgets and equal numbers of local optimizer steps. Report confidence intervals and class-wise confusion matrices to diagnose the low overall accuracy.')
p('<b>Month 2.</b> Stress client heterogeneity and participation. Measure the smallest-client and lowest-accuracy-client groups, vary class imbalance, and simulate intermittent client dropout. Test retry behavior with injected Actor failures. Benchmark serial and Ray updates under matched work across CPU counts and larger local jobs.')
p('<b>Month 3.</b> Define the privacy unit and adjacency relation, bound sensitivity, and calibrate the mechanism to explicit privacy budgets. Examine labels and repeated releases within that threat model. Compare fixed and resampled perturbations under an explicit accounting scheme. Use validation results to preselect the privacy/utility setting, then report held-out performance and client-level utility distributions.')
sub('References')
p('[1] Caldas et al. <i>LEAF, A Benchmark for Federated Settings.</i> Dataset subset and assignment requirements supplied through Canvas. ' + link('arXiv:1812.01097', 'https://arxiv.org/abs/1812.01097') + '.', 'CaptionReport')
p('[2] Ray Core documentation. ' + link('Actors', 'https://docs.ray.io/en/latest/ray-core/actors.html') + ' and ' + link('ray.kill', 'https://docs.ray.io/en/latest/ray-core/api/doc/ray.kill.html') + '.', 'CaptionReport')
p('[3] Dwork and Roth. <i>The Algorithmic Foundations of Differential Privacy.</i> Section 3.3 discusses Laplace calibration. ' + link('Author-hosted book', 'https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf') + '.', 'CaptionReport')


def decorate(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#d6dde3'))
    canvas.line(48, 752, 564, 752)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(48, 762, 'MLDS | ASSIGNMENT 3')
    canvas.drawRightString(564, 762, 'JINGLIANG GUO')
    canvas.drawString(48, 28, 'Federated EMNIST | FedAvg | Ray Core | Laplace perturbation')
    canvas.drawRightString(564, 28, str(doc.page))
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=(612, 792), rightMargin=48, leftMargin=48,
                        topMargin=53, bottomMargin=46, title='Assignment 3 - Federated Learning and Differential Privacy', author='Jingliang Guo')
doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
print(OUT)
