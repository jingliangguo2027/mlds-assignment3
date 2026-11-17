"""Build the report from saved experiment results. All numbers come from JSON."""
import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parent
D = json.loads((ROOT / 'experiment_results.json').read_text())
OUT = ROOT / 'Assignment3_Report.pdf'
BLUE = colors.HexColor('#173e62')
GRAY = colors.HexColor('#53616d')
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyReport', fontName='Helvetica', fontSize=10, leading=13.5, spaceAfter=8))
styles.add(ParagraphStyle(name='TitleReport', fontName='Helvetica-Bold', fontSize=23, leading=28, textColor=BLUE, spaceAfter=10))
styles.add(ParagraphStyle(name='SectionReport', fontName='Helvetica-Bold', fontSize=17, leading=22, textColor=BLUE, spaceAfter=12))
styles.add(ParagraphStyle(name='SubReport', fontName='Helvetica-Bold', fontSize=11.5, leading=15, textColor=BLUE, spaceBefore=6, spaceAfter=6))
styles.add(ParagraphStyle(name='CaptionReport', fontName='Helvetica', fontSize=8.5, leading=11, textColor=GRAY, spaceAfter=10))
styles.add(ParagraphStyle(name='CellReport', fontName='Helvetica', fontSize=8.8, leading=11.5))
styles.add(ParagraphStyle(name='CodeReport', fontName='Courier', fontSize=8.4, leading=12, spaceAfter=8))
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
    story.append(Spacer(1, 5))


def caption(text):
    p(text, 'CaptionReport')


def table(rows, widths):
    content = [[Paragraph(escape(str(x)), styles['CellReport']) for x in row] for row in rows]
    t = Table(content, colWidths=[x * inch for x in widths], repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e9f0f6')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LINEBELOW', (0, 0), (-1, 0), .7, BLUE),
        ('LINEBELOW', (0, 1), (-1, -1), .3, colors.HexColor('#d6dde3')),
    ]))
    story.append(t)
    story.append(Spacer(1, 9))


def page():
    story.append(PageBreak())


def pct(x):
    return f'{x:.1%}'


def link(label, url):
    return f'<link href="{url}" color="#173e62">{label}</link>'


data, tune, sel = D['data'], D['tuning'], D['selection']
seq, ray, nr, v, tests = D['sequential'], D['ray'], D['noise'], D['validation'], D['test']
cfg = sel['chosen']['config']
client = D['clients']
sizes, coverage = np.array(data['sizes']), np.array(data['coverage'])
base = next(r for r in tune['rows'] if r['config']['inputs'] == 'raw' and r['config']['lr'] == 0.01)
base_run = json.loads((ROOT / 'runs' / base['run']).read_text())
choice = str(nr['choice'])
noise_mean = {b: np.mean([r['validation']['accuracy'] for r in rows]) for b, rows in nr['records'].items()}
noise_sd = {b: np.std([r['validation']['accuracy'] for r in rows], ddof=1) for b, rows in nr['records'].items()}

p('Federated Learning<br/>and Differential Privacy', 'TitleReport')
p('Assignment 3 | Jingliang Guo | November 17, 2026', 'SubReport')
p('GitHub username: <b>jingliangguo2027</b><br/>Public repository: <b>mlds-assignment3</b><br/>' + link('github.com/jingliangguo2027/mlds-assignment3', 'https://github.com/jingliangguo2027/mlds-assignment3'))
sub('Data and model')
p(f'There are {data["clients"]} clients, {data["total"]:,} local images, and {data["test"]:,} held-out test images. Each image is 28 x 28 pixels. There are 62 classes. Each client has its own shuffled 80/20 split, using split seed 42. This gives {data["train"]:,} training and {data["validation"]:,} validation images.')
p(f'The network is Linear(784,128), ReLU, and Linear(128,62). I use cross-entropy loss, SGD, and batches of 32. The final setting uses input <b>1 - X</b>, learning rate <b>{cfg["lr"]:g}</b>, C=<b>{cfg["clients"]}%</b>, E=<b>{cfg["epochs"]}</b>, and <b>{cfg["rounds"]} rounds</b>. All training uses CPU with one PyTorch thread.')
figure('class_distribution.png', 6.05)
caption('Figure 1. Class percentages for all local data and clients 0-4, with shared axes. Class IDs follow the dataset. [1]')
p(f'Client sizes range from {sizes.min()} to {sizes.max()} images. Clients contain {coverage.min()} to {coverage.max()} classes. Some classes are absent locally, so each client still keeps all 62 model outputs.')

page()
title('Why the first experiment learned slowly')
p(f'The original pixels range from {data["pixel_min"]:g} to {data["pixel_max"]:g}, with mean {data["pixel_mean"]:.3f}. Most pixels are white background. I compared X with 1 - X. The second form makes the background zero. This changes the input representation but keeps the same image information.')
p(f'I also varied the learning rate. All points below use 100 rounds, C=10%, E=3, the same initial weights, and the same selected clients and batch seeds. Original inputs with lr=0.01 reach {pct(base["validation"]["accuracy"])} validation accuracy. The best tested input/rate pair reaches {pct(tune["chosen"]["validation"]["accuracy"])}.')
figure('optimization_diagnosis.png')
caption('Figure 2. Left: learning-rate comparison. Right: paired learning curves. Raising the rate alone does not solve the problem with the original inputs. No test results were used here.')
sub('Does a longer run help?')
rows = [['Rounds', 'Validation accuracy', 'What stayed fixed']]
for b in sel['budgets']:
    rows.append([b['rounds'], pct(b['validation']['accuracy']), f'1 - X, lr={tune["chosen"]["config"]["lr"]:g}, C=10%, E=3'])
table(rows, [1.0, 1.7, 3.95])
p('The longer run improves validation accuracy, but the gain becomes smaller. I use the best final validation result among these three budgets. These experiments support changing both the input form and the training settings. They do not prove that the chosen rate is best for every setup.')

page()
title('Part 1 | Sequential FedAvg')
p('Each round samples clients without replacement. They start from the same global weights and train only on their own images. The server averages their updated weights. A client with n training images gets weight n divided by the total training images of the selected clients. No local images are sent to another client.')
figure('sequential_curves.png')
caption(f'Figure 3. Sample-weighted training and validation loss and accuracy for the selected model over {cfg["rounds"]} rounds. The updated global model is evaluated on all client splits.')
sub('Participation and local epochs')
rows = [['Clients per round', 'Local epochs E', 'Train accuracy', 'Validation accuracy']]
for r in sel['grid']:
    rows.append([f'{r["config"]["clients"]} ({r["config"]["clients"]}%)', r['config']['epochs'], pct(r['train']['accuracy']), pct(r['validation']['accuracy'])])
table(rows, [1.8, 1.3, 1.65, 1.9])
p(f'These runs use the same rate and {cfg["rounds"]} rounds. The selected setting is C={cfg["clients"]}%, E={cfg["epochs"]}. Its validation accuracy is <b>{pct(sel["chosen"]["validation"]["accuracy"])}</b>. Its held-out test accuracy is <b>{pct(tests["sequential"]["accuracy"])}</b>, with test loss {tests["sequential"]["loss"]:.3f}. Test data are not used for training or parameter selection.')
p('More local epochs do not consistently improve validation accuracy in this grid. At C=10%, the results are close. E=1 uses fewer local steps. These are equal-round comparisons; higher C or E means more computation.')

page()
title('Part 1 | Ray Core clients')
p(f'Ray uses four clients per round, {cfg["rounds"]} rounds, E={cfg["epochs"]}, and lr={cfg["lr"]:g}. Each selected client gets a new Actor. Each Actor reserves one CPU and zero GPUs, and trains with one PyTorch thread. Actor jobs are submitted before their results are collected. Actors are killed after each batch. [2]')
figure('ray_curves.png')
caption(f'Figure 4. Ray training and validation curves. Final validation accuracy is {pct(D["ray_run"]["history"][-1]["validation"]["accuracy"])}. Held-out test accuracy is {pct(tests["ray"]["accuracy"])}.')
sub('Compare the same work in serial and Ray')
table([['Execution', 'Clients / rounds', 'Time for local updates'],
       ['Serial CPU', f'4 / {cfg["rounds"]}', f'{ray["serial_update_seconds"]:.2f} seconds'],
       ['Ray Actors', f'4 / {cfg["rounds"]}', f'{ray["ray_update_seconds"]:.2f} seconds']], [1.65, 1.8, 3.2])
p(f'The two runs use identical initial weights, client selections, and local random seeds. After {cfg["rounds"]} rounds, the maximum weight difference is <b>{ray["max_parameter_error"]:.2e}</b>. The largest validation-accuracy difference over all rounds is <b>{ray["max_validation_curve_error"]:.2e}</b>. The execution method preserves the training result.')
p('The measured times include Actor creation and cleanup, but exclude global evaluation. Ray is slower for these small jobs because process startup and data transfer cost more than the saved training time. The 10-client run does more local work per round than this four-client run, so their accuracy gap is not caused by Ray itself.')

page()
title('Part 2 | Add Laplace noise to training images')
p(f'For every training pixel, I add independent zero-centered Laplace noise with scale b. Noise is sampled once per image and kept fixed during training. The model receives 1 - (X + noise). Values are not clipped. All scales use C=10%, E={cfg["epochs"]}, lr={cfg["lr"]:g}, and {cfg["rounds"]} rounds.')
p('Training accuracy below uses the actual noisy training inputs. Validation and test inputs stay clean. I also measure clean-training accuracy separately. This makes the effect of the fixed noise easier to see.')
figure('noise_round_curves.png')
caption('Figure 5. Training and validation accuracy versus communication round, using seed 42. Each curve uses the same starting weights and client selections.')
figure('noise_final_accuracy.png')
caption('Figure 6. Left: final noisy-training, clean-training, and clean-test accuracy. Right: mean validation accuracy and sample standard deviation for seeds 42, 43, and 44. The client split stays fixed.')
p('Larger noise can make the stored training images easy to fit while hurting clean validation and test accuracy. A high noisy-training accuracy is therefore not enough to show that the model has learned useful image features.')

page()
title('Part 2 | Results and choice of noise scale')
rows = [['Scale b', 'Noisy train, seed 42', 'Clean test, seed 42', 'Mean validation', 'Sample SD']]
for b, records in nr['records'].items():
    rows.append([f'{float(b):g}', pct(records[0]['train']['accuracy']), pct(tests['noise'][b]['accuracy']), pct(noise_mean[b]), pct(noise_sd[b])])
table(rows, [.75, 1.55, 1.5, 1.55, 1.3])
p(f'I choose <b>b={nr["choice"]:g}</b>. Before checking test results, I set this rule: choose the largest positive tested scale whose mean validation accuracy is within 2 percentage points of the clean baseline. Mean validation accuracy is {pct(noise_mean["0.0"])} at b=0 and {pct(noise_mean[choice])} at the chosen scale, a change of {(noise_mean[choice]-noise_mean["0.0"])*100:+.2f} percentage points.')
p(f'The chosen seed-42 model has held-out test accuracy <b>{pct(tests["noise"][choice]["accuracy"])}</b>. b=0.1 gives mean validation accuracy {pct(noise_mean["0.1"])}. Larger scales lose more accuracy in this experiment. This choice depends on the two-point tolerance and the scales tested; three seeds do not establish a universal best value.')
sub('How much does the model depend on its stored noise?')
figure('fresh_noise_check.png')
caption('Figure 7. The same training images and labels are evaluated with their stored noise, an independent new noise draw, and no noise. Only evaluation inputs change. This is a validation experiment, not a new training run.')
fresh = v['fresh_noise_evaluation']['0.3']
p(f'At b=0.3, accuracy is {pct(fresh["stored_noise_accuracy"])} on the stored noisy images but {pct(fresh["fresh_noise_accuracy"])} on a fresh noise draw. The gap shows that fitting the stored perturbations does not mean the model is robust to new perturbations.')
sub('Privacy interpretation')
p('A larger b perturbs each pixel more. The chosen scale changes the local inputs while keeping much of the clean accuracy. This is not a numerical privacy guarantee. Such a guarantee would need a clear definition of what record or user is protected, how the release changes when that data changes, and what information is released. Labels are unchanged here. [3]')

page()
title('Look at individual clients and classes')
figure('client_baseline.png')
caption('Figure 8. Left: each dot is one client; the dashed line marks equal accuracy. The baseline predicts that client\'s most common training label. It does not use validation labels to choose a class. Right: recall for the twelve hardest validation classes. This is the Part 1 non-standard plot.')
p(f'FedAvg beats the local majority baseline for <b>{client["baseline_wins"]}/100 clients</b>. The 10th percentile of client accuracy is {pct(client["accuracy_p10"])}; the lowest is {pct(client["accuracy_min"])}. Average recall across the 62 classes is {pct(client["macro_recall"])}; this gives each class equal weight. These checks show performance that the overall average can hide.')
p('The most common validation confusions (true class to predicted class) are ' + '; '.join(f'{r["true"]} to {r["predicted"]}: {r["count"]} images' for r in client['common_confusions'][:3]) + '. Class IDs are used because the supplied assignment does not provide a character-name mapping.')
confusion = np.array(client['confusion'])
support = confusion.sum(1)
recall = np.divide(confusion.diagonal(), support, out=np.zeros(62), where=support > 0)
weak = int(np.argmin(recall))
p(f'Class {weak} has {confusion[weak,weak]} correct predictions out of {support[weak]} validation images. Several hard classes have very few validation examples. Overall accuracy has improved, but the model still handles some rare classes poorly.')
figure('noise_client_effect.png')
caption(f'Figure 9. Change in each client\'s validation accuracy at the selected b={nr["choice"]:g}, relative to b=0, using seed 42. The left panel compares the change with client size. The right panel counts clients by change. This is the Part 2 non-standard plot.')
impact=nr['client_effect']
p(f'Noise improves accuracy for {impact["clients_better"]} clients, reduces it for {impact["clients_worse"]}, and leaves {impact["clients_equal"]} unchanged. The median change is {impact["median_pp"]:+.2f} percentage points. The worst change is {impact["worst_pp"]:+.2f} points. A small average loss can still matter to some users.')

page()
title('Validation beyond checking the code runs')
sub('Break the image-label relationship')
sh=v['shuffled_label_control']
p(f'I shuffle training labels within each client. This keeps every client\'s sample count and label distribution unchanged. Validation labels stay correct. At 100 rounds, real labels give <b>{pct(sh["real_label_validation"]["accuracy"])}</b> validation accuracy; shuffled labels give <b>{pct(sh["validation"]["accuracy"])}</b>. The improvement therefore depends on matching images to labels, rather than only learning class frequencies.')
sub('Can the network fit a small known task?')
p(f'On 16 fixed training images, loss falls from {v["tiny_before"]["loss"]:.3f} to {v["tiny_after"]["loss"]:.3f}, and accuracy rises from {pct(v["tiny_before"]["accuracy"])} to {pct(v["tiny_after"]["accuracy"])} after 500 local epochs. This rules out a complete failure of the network or optimizer. It does not show that a full federated run is well tuned; the input/rate and round-budget experiments address that separately.')
sub('Repeat training and test extreme noise')
p(f'The main noise comparison repeats every scale with seeds 42, 43, and 44. Initial weights, batch order, client choices, and noise draws vary; the local split stays fixed. The clean validation mean is {pct(noise_mean["0.0"])} with sample SD {pct(noise_sd["0.0"])}. These repeats show variation on this split, not uncertainty across all possible client datasets.')
stress=v['strong_noise']
p(f'At b=3, {pct(stress["outside_fraction"])} of perturbed pixels fall outside [0,1]. With the selected learning rate, a non-finite update appears at round {stress["failed_round"]}. The code rejects that round before changing the global model. Reducing the rate to 0.01 lets all {cfg["rounds"]} rounds finish, but validation accuracy is only {pct(v["strong_noise_lower_rate"]["validation"]["accuracy"])}. Thus the clean-data rate is unsafe for this extreme noise level, and a stable smaller rate still gives poor utility.')
sub('Basic algorithm limits')
rows=[['Check', 'Expected behavior', 'Observed maximum weight error'],
      ['Unequal client sizes', 'Use training counts, not an equal mean',f'{v["aggregation_error"]:.2e}'],
      ['Single client', 'FedAvg equals that client update',f'{v["single_client_error"]:.2e}'],
      ['Zero learning rate', 'Weights remain unchanged',f'{v["zero_lr_error"]:.2e}']]
table(rows,[1.5,3.5,1.65])
p('These last checks test the update equations. The shuffled-label control, input/rate comparisons, longer runs, seed repeats, and per-client plots test whether the algorithm learns the intended task.')

page()
title('Edge cases and how they were tested')
sub('Part 1: uneven data, few CPUs, and client failure')
p(f'<b>Unequal sizes and missing classes.</b> Clients have {sizes.min()}-{sizes.max()} images and {coverage.min()}-{coverage.max()} observed classes. The server uses training-sample weights. The two-client calculation uses {v["unequal_client_counts"][0]} and {v["unequal_client_counts"][1]} training images. The model always has 62 outputs. Figure 8 checks clients and classes separately.')
p(f'<b>Small splits and leakage.</b> The loader rejects a client with fewer than two images. It floors 80% for training and uses the remainder for validation. Saved source indices have {v["split_overlap"]} train/validation overlaps across all 100 clients. The test archive is opened only after the settings and noise choice are saved.')
one=v['one_cpu']; failure=v['actor_failure']
p(f'<b>Only one CPU.</b> A real Ray cluster limited to one CPU processes four selected clients in batches {one["batches"]}. Its update matches serial execution with maximum error {one["max_parameter_error"]:.2e}. The reserved CPU is available again after cleanup.')
p(f'<b>An Actor fails after an earlier batch succeeds.</b> I deliberately raise an error in the second client on that one-CPU cluster. The round raises an exception. The global model changes by {failure["unchanged_model_error"]:.2e}, and the CPU is released. Retrying the complete round succeeds and matches the serial reference with error {failure["successful_retry_error"]:.2e}. A partial group of updates is never committed.')
sub('Part 2: zero, invalid, and strong noise')
noise_check=v['laplace']
p('<b>Zero and invalid scales.</b> b=0 preserves every training pixel exactly. Negative, NaN, and infinite scales raise ValueError. The zero-noise model is also the clean reference in every scale comparison.')
p(f'<b>Check the sampled distribution.</b> At b=0.3, sampled noise has mean {noise_check["mean"]:.4f}, mean absolute value {noise_check["mean_absolute"]:.4f}, and variance {noise_check["variance"]:.4f}. The expected values are 0, 0.3, and {noise_check["expected_variance"]:.2f}. The mean/absolute-mean tolerance is 2% of b; variance tolerance is 4% of its expected value.')
p('<b>Pixels outside the original range.</b> I retain them instead of clipping them, so the training data follow the specified additive mechanism. Figures 5-7 and the b=3 stress run measure the resulting loss of utility. Figure 9 checks who loses accuracy at the selected scale.')
p(f'<b>Non-finite updates.</b> Aggregation checks the candidate weights before loading them. At b=3, the failed round changes the global model by {stress["model_change_on_rejection"]:.2e}. The run stops and reports the error; it does not silently continue with invalid weights.')

page()
title('How to reproduce the results')
p('Clone the public repository. Get Assignment3-data.zip from Canvas and place it in reports/, next to hw3.ipynb. From the repository root, run:')
p('python -m pip install -r requirements.txt<br/>python reports/run_experiments.py all<br/>python reports/build_report.py','CodeReport')
p('The all command runs tuning, the C/E grid, matched serial/Ray training, noise experiments, validation controls, and final test evaluation. It then exports the plots and experiment_results.json. Each individual run is saved in reports/runs/. A cached run is reused only when its model checkpoint and matching code/configuration exist.')
p('You can also open reports/hw3.ipynb and run its cells in order. The notebook uses the same functions. Section 2 runs tuning, Section 3 runs Ray, Section 4 runs noise experiments, Section 5 runs the additional validations, and Section 6 evaluates the test set and makes the figures. Input/rate trials do not load test data.')
env=seq['environment']
p(f'The measured environment is Python {env["python"]}, PyTorch {env["torch"]}, NumPy {env["numpy"]}, and Ray 2.60.0 on an Apple M5 computer. Training uses CPU and one PyTorch thread per process. The main Ray run reserves four CPUs; the resource/failure test reserves one. Exact timings depend on the computer. GPU or other library versions can change floating-point results.')
sub('If I had three more months')
p('<b>Month 1:</b> Repeat across at least ten seeds and several new per-client splits. Check whether E=1 still matches larger E. Test learning-rate decay and early stopping using validation data. Compare equal numbers of local optimizer steps, not only equal rounds.')
p('<b>Month 2:</b> Inspect the hardest classes and weakest clients. Simulate repeated client dropout and delayed updates. Compare serial and Ray timing on larger local jobs and several CPU counts. Test whether a retry policy remains correct after repeated failures.')
p('<b>Month 3:</b> Define exactly which record or user must be protected. Bound sensitivity and choose explicit privacy budgets. Compare fixed and resampled noise under the appropriate privacy accounting. Measure both average accuracy and the losses experienced by individual clients.')
sub('References')
p('[1] Caldas et al. LEAF: A Benchmark for Federated Settings. ' + link('arXiv:1812.01097', 'https://arxiv.org/abs/1812.01097') + '. Data and task supplied through Canvas.', 'CaptionReport')
p('[2] Ray Core documentation: ' + link('Actors', 'https://docs.ray.io/en/latest/ray-core/actors.html') + ' and ' + link('ray.kill', 'https://docs.ray.io/en/latest/ray-core/api/doc/ray.kill.html') + '.', 'CaptionReport')
p('[3] Dwork and Roth. The Algorithmic Foundations of Differential Privacy, Section 3.3. ' + link('Book', 'https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf') + '.', 'CaptionReport')


def decorate(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#d6dde3'))
    canvas.line(48, 752, 564, 752)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(48, 762, 'MLDS | ASSIGNMENT 3')
    canvas.drawRightString(564, 762, 'JINGLIANG GUO')
    canvas.drawString(48, 28, 'Federated EMNIST | FedAvg | Ray Core | Laplace noise')
    canvas.drawRightString(564, 28, str(doc.page))
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=(612, 792), rightMargin=48, leftMargin=48,
                        topMargin=53, bottomMargin=46,
                        title='Assignment 3 - Federated Learning and Differential Privacy', author='Jingliang Guo')
doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
print(OUT)
