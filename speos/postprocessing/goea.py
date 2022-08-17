import pandas as pd
from coregenes.postprocessing.lightgoea.light_goea import GOEA
import numpy as np
import os


class GOEA_Study:
    def __init__(self, base_path="./data/genescf/geneSCF-master-source-v1.1-p2/class/lib/db/goa_human/",
                filename_scheme = "GO_{}_sym.txt",
                task_dict = None):
        self.base_path = base_path
        self.filename_scheme = filename_scheme
        if task_dict is None:
            self.task_dict = {"biological process": "BP",
                              "molecular function": "MF",
                              "cellular component": "CC"}
        else:
            self.task_dict = task_dict
        self.go2name = None
        self.go2symbol = None

    def reset(self):
        """ Use this if you want to re-use the same goea object for another task without re-initiliazing it"""
        self.go2name = None
        self.go2symbol = None

    def analyze(self, results: list, background: set, task="biological process") -> pd.DataFrame:
        """ see if results are enriched in background for given task. Both results and background should contain HGNC symbols
        
        implemented tasks: "biological process", "molecular function", "cellular component"
        """
        if self.go2name is None or self.go2symbol is None:
            self.get_datastructures(task)

        # purge symbols that are not in the background set from the ontology
        for go_term, symbols in self.go2symbol.items():
            blacklist = []
            for symbol in symbols:
                if symbol not in background:
                    blacklist.append(symbol)

            for to_delete in blacklist:
                symbols.remove(to_delete)

            self.go2symbol[go_term] = symbols

        df = GOEA(np.asarray(results), self.go2symbol, fdr_thresh=0.05)

        descriptions = [self.go2name[go] for go in df.index]
        df["description"] = [description.capitalize() for description in descriptions]

        df = self.calculate_goea_statistics(df, results, background, self.go2symbol)

        return df

    def get_datastructures(self, task=None, term_df=None):
        if term_df is None:
            path = os.path.join(self.base_path, self.filename_scheme.format(self.task_dict[task]))
            go_terms = pd.read_csv(path, header=0, sep="\t", names=["term", "symbol"])
        else:
            if task is None:
                raise ValueError("Either task or term_df must be set (not None)")
            go_terms = term_df
        self.go2name = {line.split("~")[0]: line.split("~")[1] for line in go_terms["term"]}
        self.go2symbol = {line[0].split("~")[0]: list(line[1][:-1].split(",")) for i, line in go_terms.iterrows()}

    def set_term_description(self, term_description: dict):
        self.go2name = term_description
    
    def set_term_symbol(self, term_symbol: dict):
        self.go2symbol = term_symbol

    def calculate_goea_statistics(self, df, results, background_symbols_set, go2symbol) -> pd.DataFrame:
        """calculates the necessary stats such as enrichment, expected, log_q etc and appends it to the df"""

        fraction = len(results) / len(background_symbols_set)
        observed_list = []
        total_list = []
        expected_list = []
        enrichment_list = []
        for term in df.index:
            observed = 0
            for symbol in go2symbol[term]:
                if symbol in set(results):
                    observed += 1
            total = len(go2symbol[term])
            expected = total * fraction
            enrichment = observed / expected
            observed_list.append(observed)
            total_list.append(total)
            expected_list.append(expected)
            enrichment_list.append(enrichment)

        df["observed"] = observed_list
        df["total"] = total_list
        df["expected"] = expected_list
        df["enrichment"] = enrichment_list
        df["log_q"] = np.log10(df["fdr_q_value"]) * -1

        return df

    def plot(self, df, path):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import textwrap
        import numpy as np
        from matplotlib.lines import Line2D

        fig, ax = plt.subplots(figsize=(6, 0.45*len(df.index)))
        cmap = mpl.cm.coolwarm
        norm = mpl.colors.Normalize(vmin=df.enrichment.min(), vmax=df.enrichment.max())
        mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

        simple_significant = np.log10(0.05) * -1
        highly_significant = np.log10(0.001) * -1

        max_size = 25
        max_size_factor = 1
        count = np.asarray(df["observed"][::-1])
        count_normalized = (count / count.max()) * max_size * max_size_factor

        plt.axvline(x=simple_significant, color='lightgrey', linestyle='--', linewidth=1, zorder=-1)
        plt.axvline(x=highly_significant, color='lightgrey', linestyle=':', linewidth=1, zorder=-1)
        plt.scatter(df["log_q"][::-1], df["description"][::-1].tolist(), color=mapper.to_rgba(df["enrichment"][::-1]), s=count_normalized**2, zorder=2)
        ax.set_yticklabels([textwrap.fill(e, 30) for e in df["description"]][::-1])
        fig.colorbar(mapper, orientation='horizontal', location="top", label="Fold Enrichment", pad=0.005, ax=ax)
        ax.set_xlabel(r"$-\log(q)$")

        legend_elements = [Line2D([0], [0], marker='o', color='grey', label="{} Genes".format(int(count.min())),
                                  markerfacecolor='w', markersize=count_normalized.min(), linestyle=''),
                           Line2D([0], [0], marker='o', color='grey', label="{} Genes".format(int(count.mean())),
                                  markerfacecolor='w', markersize=int(count_normalized.mean()), linestyle=''),
                           Line2D([0], [0], marker='o', color='grey', label="{} Genes".format(int(count.max())),
                                  markerfacecolor='w', markersize=count_normalized.max(), linestyle='')
                           ]

        ax.legend(handles=legend_elements, loc='upper left', labelspacing=2, borderpad=1.5)
        plt.ylim([-0.5, len(df["description"])])
        plt.xlim([0, df["log_q"].max()+1])
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight')
