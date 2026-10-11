"""Update only affected current single-panel exports; leave PPTs untouched."""
import argparse
import importlib.util
from pathlib import Path
import shutil
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = Path(r'D:\Published paper\high-order-topology\CSF_prepartion\Figure')
CSF = ROOT / 'Chaos, Solitons & Fractals'
V2 = ROOT / 'tables/computational_closure_v2'
sys.path.insert(0, str(ROOT / 'code'))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(phase):
    story = load(ROOT / 'code/run_story_figures_v3.py', 'current_story_null100')
    story.SOURCE_DIR = CSF / 'source_data'
    story.export_publication_figure = lambda fig, *args, **kwargs: plt.close(fig)
    if phase == 'early':
        m3 = load(FIG / 'Figure.3/plot_figure_3d_compact.py', 'null100_3d')
        m3.main()
        m6 = load(FIG / 'Figure.6/export_figure6_abc_without_letters.py', 'null100_6ab')
        m6.PANEL_EXPORTS = {key: value for key, value in m6.PANEL_EXPORTS.items() if key in (0, 1)}
        m6.main()
        shutil.copy2(FIG / 'Figure.6/Figure_6_source_data.csv', CSF / 'source_data/Figure_6_source_data.csv')
        load(FIG / 'Figure.6/export_figure6d_adjusted.py', 'null100_6d').export_panel()
        load(FIG / 'Figure.S6/plot_figure_s6_revised.py', 'null100_s6').main()
        story.supplementary_6()
        story.supplementary_9()
        load(FIG / 'Figure.S9/plot_figure_s9_revised.py', 'null100_s9').main()
        # Preserve the unrelated burst data while replacing the adjusted contrast rows.
        path = CSF / 'source_data/Figure_3_source_data.csv'
        previous = pd.read_csv(path)
        print('Figure 3 source datasets:', previous.dataset.unique(), flush=True)
        non_adjusted = previous[previous.adjustment_tier.isna()].copy()
        current = m3.load_effects()
        adjusted_names = previous.loc[previous.adjustment_tier.notna(), 'dataset'].unique()
        if len(adjusted_names) != 1:
            raise ValueError(f'Ambiguous adjusted source data: {adjusted_names}')
        current.insert(0, 'dataset', adjusted_names[0])
        pd.concat([non_adjusted, current], ignore_index=True).to_csv(path, index=False)
    elif phase == 'late':
        story.supplementary_10()
        load(FIG / 'Figure.S10/plot_figure_s10_revised.py', 'null100_s10').main()
    print(f'Completed {phase} current-panel exports', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['early', 'late'])
    main(parser.parse_args().phase)
