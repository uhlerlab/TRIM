from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import os

def investigate_corner_genes(diffexp_df, 
                             colname, 
                             cellType, 
                             lfc_type,
                             title,
                             de_type):

    lfc_folder = f'{lfc_type}_by_gradient'

    if not os.path.exists(f'/home/che/TRIM/git/tcr/figures/saliency/{de_type}/{cellType}/{lfc_folder}/{colname}'):
        os.makedirs(f'/home/che/TRIM/git/tcr/figures/saliency/{de_type}/{cellType}/{lfc_folder}/{colname}')
    
    colors = np.where(diffexp_df[colname] == 'Yes', 'red', 'lightgrey')
    # Create scatter plot
    fig, ax1 = plt.subplots()
    ax1.scatter(diffexp_df['salient_genes'], diffexp_df[lfc_type], s=5, alpha=0.8, c=colors)
    ax1.set(xlabel='Salient Genes', ylabel='Log Fold Change', title=title)
    legend = ax1.legend(handles=[
        mpatches.Patch(color='red', label=f'{colname} = Yes'),
        mpatches.Patch(color='lightgrey', label=f'{colname} = No')
    ], loc='upper left', bbox_to_anchor=(1, 1), title='Gene Status')
    plt.tight_layout()
    output_plot_path = f'/home/che/TRIM/git/tcr/figures/saliency/{de_type}/{cellType}/{lfc_folder}/{colname}/salient_genes_scatter_with_legend.png'
    fig.savefig(output_plot_path, bbox_extra_artists=(legend,), bbox_inches='tight')

    ## (1.2) Take a deeper look at the genes picked up by high-grads and high-lfc
    gene_list = diffexp_df[diffexp_df[colname] == 'Yes'].index.tolist()
    print(len(gene_list))
    mean_expression_df = diffexp_df.loc[gene_list]
    mean_expression_df.to_csv(f'/home/che/TRIM/git/tcr/figures/saliency/{de_type}/{cellType}/{lfc_folder}/{colname}/mean_gene_expression.csv', index=True)
    
    # pathway analysis
    # genes_for_analysis = diffexp_df[diffexp_df[colname] == 'Yes'].index.tolist()
    # # save genes_for_analysis
    # with open(f'tcr/git/tcr/figures_saliency/{cellType}/{colname}/genes_for_analysis.txt', 'w') as f:
    #     for item in genes_for_analysis:
    #         f.write("%s\n" % item)
    # all_genes = diffexp_df.index.tolist()

    # try:
    #     enr = gp.enrichr(
    #         gene_list=genes_for_analysis,  # List of genes
    #         background=all_genes,  # Background genes
    #         gene_sets='KEGG_2021_Human',  # You can choose other gene sets like GO_Biological_Process, Reactome, etc.
    #         organism='Human',  # 'Human', 'Mouse', etc.
    #         outdir=f'tcr/git/tcr/figures_saliency/{cellType}/{colname}/gsea_results'
    #     )
    #     ax = barplot(enr.res2d, title='KEGG_2021_Human', figsize=(4, 5), color='darkred')
    #     enr.results.to_csv(f'tcr/git/tcr/figures_saliency/{cellType}/{colname}/gsea_results/enrichr_results.csv')
    # except Exception as e:
    #     print(f"Error performing Enrichr analysis: {e}")
