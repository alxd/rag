import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from upsetplot import UpSet, from_indicators
import numpy as np
import os
import warnings
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
from collections import defaultdict
import re
import html
import traceback
import textwrap
import win32com.client
import subprocess
import random
import json
import openai
try:
    import seaborn as sns
except ImportError:
    sns = None
try:
    from mistralai import Mistral
except ImportError:
    Mistral = None
try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.table import _Cell, Table

# Common suffixes for substring grouping (except suffixes)
common_suffixes = [
    'ation','ption', 'ment', 'ness', 'sion', 'tion', 'ing', 'ed', 'ly', 'er', 'est', 'ful', 'less', 'able', 'ible', 'ous', 'ive', 'al', 'ic', 'ant', 'ent', 'ism', 'ist', 'ity', 'ty', 'en', 'ize', 'ise', 'ward', 'wise'
]

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# RAG Consistency Analyzer for semantic similarity and exact matching
class RAGConsistencyAnalyzer:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            from scipy.optimize import linear_sum_assignment
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.data = []
        except ImportError as e:
            print(f"Warning: Could not import required libraries for semantic analysis: {e}")
            self.model = None
            self.model_name = model_name
            self.data = []
    
    def parse_concepts(self, concept_string):
        """Extract individual concepts from the string"""
        if not concept_string:
            return []
        
        # Split by common patterns and clean
        concepts = []
        # Handle different formats in your data
        parts = concept_string.replace('\n', ' ').split()
        
        # Simple extraction - you may need to refine based on exact format
        current_concept = []
        for part in parts:
            if any(greek in part for greek in ['ἑκούσιον', 'προαίρεσις', 'φρόνησις', 'ἀρετή']) or \
               any(term in part for term in ['Voluntary', 'Moral', 'Practical', 'Virtue', 'Justice']):
                if current_concept:
                    concepts.append(' '.join(current_concept))
                    current_concept = []
            current_concept.append(part)
        
        if current_concept:
            concepts.append(' '.join(current_concept))
        
        return concepts[:10]  # Limit to 10 concepts
    
    def compute_concept_consistency(self, concepts_list1, concepts_list2):
        """Compute semantic similarity between two concept lists"""
        if not self.model or not concepts_list1 or not concepts_list2:
            return 0.0
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            from scipy.optimize import linear_sum_assignment
            import numpy as np
            
            # Compute embeddings
            embeds1 = self.model.encode(concepts_list1)
            embeds2 = self.model.encode(concepts_list2)
            
            # Hungarian algorithm for optimal concept matching
            similarity_matrix = cosine_similarity(embeds1, embeds2)
            row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
            matched_similarities = similarity_matrix[row_ind, col_ind]
            
            return np.mean(matched_similarities)
        except Exception as e:
            print(f"Error computing semantic similarity: {e}")
            return 0.0
    
    def exact_concept_overlap(self, concepts_list1, concepts_list2):
        """Compute Jaccard similarity for exact concept matching"""
        if not concepts_list1 or not concepts_list2:
            return 0.0
            
        # Normalize concepts for comparison (remove Greek text, punctuation)
        def normalize_concept(concept):
            # Extract main English term
            concept = concept.split('(')[0].strip()
            return concept.lower()
        
        set1 = {normalize_concept(c) for c in concepts_list1}
        set2 = {normalize_concept(c) for c in concepts_list2}
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def add_result(self, temperature, top_p, top_k, bm25_weight, concepts_text):
        """Add a single experimental result"""
        concepts = self.parse_concepts(concepts_text)
        self.data.append({
            'temperature': temperature,
            'top_p': top_p, 
            'top_k': top_k,
            'bm25_weight': bm25_weight,
            'concepts': concepts
        })
    
    def analyze_single_parameter_stability(self, parameter_name):
        """Analyze stability within variations of a single parameter"""
        # Group by the parameter being varied
        parameter_groups = {}
        for row in self.data:
            param_value = row[parameter_name]
            if param_value not in parameter_groups:
                parameter_groups[param_value] = []
            parameter_groups[param_value].append(row)
        
        # For each parameter value, compute consistency with other parameter values
        group_consistencies = []
        
        for param_value, group_data in parameter_groups.items():
            if len(group_data) < 2:
                continue
            
            # Compute pairwise similarities within this parameter value group
            similarities = []
            exact_overlaps = []
            
            for i in range(len(group_data)):
                for j in range(i + 1, len(group_data)):
                    concepts1 = group_data[i]['concepts']
                    concepts2 = group_data[j]['concepts']
                    
                    semantic_sim = self.compute_concept_consistency(concepts1, concepts2)
                    exact_overlap = self.exact_concept_overlap(concepts1, concepts2)
                    
                    similarities.append(semantic_sim)
                    exact_overlaps.append(exact_overlap)
            
            if similarities:
                group_consistencies.append({
                    'parameter_value': param_value,
                    'semantic_similarity_mean': np.mean(similarities),
                    'semantic_similarity_std': np.std(similarities),
                    'exact_overlap_mean': np.mean(exact_overlaps),
                    'exact_overlap_std': np.std(exact_overlaps),
                    'n_comparisons': len(similarities)
                })
        
        # Also compute cross-parameter-value consistency (how consistent is this parameter across different values)
        cross_value_similarities = []
        cross_value_exact_overlaps = []
        
        param_values = list(parameter_groups.keys())
        for i in range(len(param_values)):
            for j in range(i + 1, len(param_values)):
                val1 = param_values[i]
                val2 = param_values[j]
                
                # Compare all combinations between these two parameter values
                for row1 in parameter_groups[val1]:
                    for row2 in parameter_groups[val2]:
                        concepts1 = row1['concepts']
                        concepts2 = row2['concepts']
                        
                        semantic_sim = self.compute_concept_consistency(concepts1, concepts2)
                        exact_overlap = self.exact_concept_overlap(concepts1, concepts2)
                        
                        cross_value_similarities.append(semantic_sim)
                        cross_value_exact_overlaps.append(exact_overlap)
        
        return {
            'parameter': parameter_name,
            'parameter_values': param_values,
            'group_results': group_consistencies,
            'cross_value_semantic_mean': np.mean(cross_value_similarities) if cross_value_similarities else 0.0,
            'cross_value_exact_mean': np.mean(cross_value_exact_overlaps) if cross_value_exact_overlaps else 0.0,
            'overall_semantic_mean': np.mean([g['semantic_similarity_mean'] for g in group_consistencies]) if group_consistencies else 0.0,
            'overall_exact_mean': np.mean([g['exact_overlap_mean'] for g in group_consistencies]) if group_consistencies else 0.0,
            'total_comparisons': len(cross_value_similarities)
        }
    
    def analyze_cross_parameter_stability(self):
        """Analyze stability across all parameter combinations"""
        all_similarities = []
        all_exact_overlaps = []
        
        # All pairwise comparisons
        for i in range(len(self.data)):
            for j in range(i + 1, len(self.data)):
                concepts1 = self.data[i]['concepts']
                concepts2 = self.data[j]['concepts']
                
                semantic_sim = self.compute_concept_consistency(concepts1, concepts2)
                exact_overlap = self.exact_concept_overlap(concepts1, concepts2)
                
                all_similarities.append(semantic_sim)
                all_exact_overlaps.append(exact_overlap)
        
        return {
            'overall_semantic_mean': np.mean(all_similarities) if all_similarities else 0.0,
            'overall_semantic_std': np.std(all_similarities) if all_similarities else 0.0,
            'overall_exact_mean': np.mean(all_exact_overlaps) if all_exact_overlaps else 0.0,
            'overall_exact_std': np.std(all_exact_overlaps) if all_exact_overlaps else 0.0,
            'total_comparisons': len(all_similarities)
        }
    
    def generate_stability_report(self, run_within_param=True, run_cross_param=True, run_sensitivity=True):
        """Generate comprehensive stability analysis"""
        if run_within_param or run_cross_param or run_sensitivity:
            print("=== RAG CONCEPT CONSISTENCY ANALYSIS ===\n")
        
        results = {}
        
        # Analyze each parameter individually (for within-parameter analysis)
        # Also run if sensitivity is requested, as it depends on individual_parameters
        if run_within_param or run_sensitivity:
            parameters = ['temperature', 'top_p', 'top_k', 'bm25_weight']
            individual_results = {}
            
            for param in parameters:
                print(f"--- {param.upper()} STABILITY ---")
                result = self.analyze_single_parameter_stability(param)
                individual_results[param] = result
                print(f"Average Semantic Consistency: {result['overall_semantic_mean']:.3f}")
                print(f"Average Exact Overlap: {result['overall_exact_mean']:.3f}")
                print(f"Number of parameter groups: {len(result['group_results'])}")
                print()
            
            results['individual_parameters'] = individual_results
        
        # Overall cross-parameter analysis
        if run_cross_param:
            print("--- OVERALL CROSS-PARAMETER STABILITY ---")
            overall = self.analyze_cross_parameter_stability()
            results['cross_parameter'] = overall
            print(f"Semantic Similarity: {overall['overall_semantic_mean']:.3f} ± {overall['overall_semantic_std']:.3f}")
            print(f"Exact Overlap: {overall['overall_exact_mean']:.3f} ± {overall['overall_exact_std']:.3f}")
            print(f"Total comparisons: {overall['total_comparisons']}")
        
        return results
    
    def generate_semantic_similarity_heatmap(self, output_dir, folder_name):
        """Generate semantic similarity matrix heatmap for all concepts in this folder"""
        if not self.model or not self.data:
            return None
            
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # Collect all unique concepts from this folder
            all_concepts = []
            for row in self.data:
                all_concepts.extend(row['concepts'])
            
            # Remove duplicates while preserving order
            unique_concepts = list(dict.fromkeys(all_concepts))
            
            if len(unique_concepts) < 2:
                print(f"[SEMANTIC VIZ] Not enough concepts for heatmap in {folder_name}")
                return None
            
            # Limit to reasonable number for visualization
            if len(unique_concepts) > 50:
                # Take most frequent concepts
                concept_counts = {}
                for row in self.data:
                    for concept in row['concepts']:
                        concept_counts[concept] = concept_counts.get(concept, 0) + 1
                unique_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:50]
                unique_concepts = [concept for concept, count in unique_concepts]
            
            # Compute embeddings for all unique concepts
            embeddings = self.model.encode(unique_concepts)
            
            # Compute similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Create heatmap
            plt.figure(figsize=(max(12, len(unique_concepts) * 0.4), max(10, len(unique_concepts) * 0.4)))
            
            # Truncate concept names for better display
            display_concepts = [concept[:30] + '...' if len(concept) > 30 else concept for concept in unique_concepts]
            
            sns.heatmap(similarity_matrix, 
                       annot=True, 
                       fmt='.2f',
                       cmap='viridis',
                       xticklabels=display_concepts,
                       yticklabels=display_concepts,
                       cbar_kws={'label': 'Semantic Similarity'})
            
            plt.title(f'Semantic Similarity Matrix - {folder_name}\n(Higher values = more similar concepts)', 
                     fontsize=14, pad=20)
            plt.xlabel('Concepts', fontsize=12)
            plt.ylabel('Concepts', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save the plot
            heatmap_path = os.path.join(output_dir, f"{folder_name}_semantic_similarity_heatmap.png")
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[SEMANTIC VIZ] Created similarity heatmap: {heatmap_path}")
            return heatmap_path
            
        except Exception as e:
            print(f"[SEMANTIC VIZ ERROR] Failed to create similarity heatmap: {e}")
            return None
    
    def generate_tsne_plot(self, output_dir, folder_name):
        """Generate t-SNE plot for concept clustering visualization"""
        if not self.model or not self.data:
            return None
            
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Collect all unique concepts from this folder
            all_concepts = []
            for row in self.data:
                all_concepts.extend(row['concepts'])
            
            # Remove duplicates while preserving order
            unique_concepts = list(dict.fromkeys(all_concepts))
            
            if len(unique_concepts) < 3:
                print(f"[SEMANTIC VIZ] Not enough concepts for t-SNE in {folder_name}")
                return None
            
            # Limit to reasonable number for visualization
            if len(unique_concepts) > 100:
                # Take most frequent concepts
                concept_counts = {}
                for row in self.data:
                    for concept in row['concepts']:
                        concept_counts[concept] = concept_counts.get(concept, 0) + 1
                unique_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:100]
                unique_concepts = [concept for concept, count in unique_concepts]
            
            # Compute embeddings for all unique concepts
            embeddings = self.model.encode(unique_concepts)
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(unique_concepts)-1))
            embeddings_2d = tsne.fit_transform(embeddings)
            
            # Create scatter plot
            plt.figure(figsize=(12, 10))
            
            # Color points by frequency
            concept_counts = {}
            for row in self.data:
                for concept in row['concepts']:
                    concept_counts[concept] = concept_counts.get(concept, 0) + 1
            
            colors = [concept_counts.get(concept, 1) for concept in unique_concepts]
            
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                c=colors, cmap='viridis', alpha=0.7, s=100)
            
            # Add concept labels
            for i, concept in enumerate(unique_concepts):
                # Truncate long concept names
                display_concept = concept[:20] + '...' if len(concept) > 20 else concept
                plt.annotate(display_concept, 
                           (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
            
            plt.colorbar(scatter, label='Concept Frequency')
            plt.title(f't-SNE Concept Clustering - {folder_name}\n(Closer points = more similar concepts)', 
                     fontsize=14, pad=20)
            plt.xlabel('t-SNE Dimension 1', fontsize=12)
            plt.ylabel('t-SNE Dimension 2', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            tsne_path = os.path.join(output_dir, f"{folder_name}_tsne_clustering.png")
            plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[SEMANTIC VIZ] Created t-SNE plot: {tsne_path}")
            return tsne_path
            
        except Exception as e:
            print(f"[SEMANTIC VIZ ERROR] Failed to create t-SNE plot: {e}")
            return None

    def generate_cross_author_analysis(self, output_dir, current_folder, valid_folders, folder_concepts,
                                       tsne_font_size=5, tsne_n_components=2, tsne_color_palette='Set3',
                                       tsne_proximity_threshold=None, tsne_show_labels=True):
        """Generate cross-author semantic similarity analysis with heatmaps, 2D embeddings, and network graphs
        
        Args:
            output_dir: Output directory for visualizations
            current_folder: Current folder name
            valid_folders: List of valid folder paths
            folder_concepts: Dictionary mapping folders to concepts
            tsne_font_size: Font size for t-SNE labels
            tsne_n_components: Number of dimensions (2 or 3)
            tsne_color_palette: Color palette name
            tsne_proximity_threshold: Proximity threshold for grouping (None to disable)
            tsne_show_labels: Whether to show concept labels
        """
        if not self.model:
            return None
            
        try:
            import os
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.manifold import TSNE
            from sklearn.metrics.pairwise import cosine_similarity
            import networkx as nx
            
            # Collect all concepts and their embeddings from all folders
            all_concepts_data = []
            folder_names = []
            
            print(f"[CROSS-AUTHOR] Starting cross-author analysis...")
            print(f"[CROSS-AUTHOR] Valid folders: {[os.path.basename(f) for f in valid_folders]}")
            print(f"[CROSS-AUTHOR] Folder concepts keys: {list(folder_concepts.keys())}")
            
            for folder_path in valid_folders:
                # Check if folder_path is directly in folder_concepts (full path)
                if folder_path in folder_concepts:
                    concepts = list(folder_concepts[folder_path])  # Convert set to list
                    folder_name = os.path.basename(folder_path)
                    print(f"[CROSS-AUTHOR] Found {len(concepts)} concepts for {folder_name} (full path match)")
                else:
                    # Try with folder name only
                    folder_name = os.path.basename(folder_path)
                    if folder_name in folder_concepts:
                        concepts = list(folder_concepts[folder_name])  # Convert set to list
                        print(f"[CROSS-AUTHOR] Found {len(concepts)} concepts for {folder_name} (name match)")
                    else:
                        print(f"[CROSS-AUTHOR] Folder {folder_name} not found in folder_concepts")
                        print(f"[CROSS-AUTHOR] Available keys: {list(folder_concepts.keys())}")
                        continue
                
                if concepts:
                    print(f"[CROSS-AUTHOR] Processing {folder_name} with {len(concepts)} concepts")
                    print(f"[CROSS-AUTHOR] Sample concepts for {folder_name}: {concepts[:5]}")
                    # Get embeddings for concepts in this folder
                    embeddings = self.model.encode(concepts)
                    for concept, embedding in zip(concepts, embeddings):
                        all_concepts_data.append({
                            'concept': concept,
                            'author': folder_name,
                            'embedding': embedding
                        })
                    folder_names.append(folder_name)
                else:
                    print(f"[CROSS-AUTHOR] No concepts found for {folder_name}")
            
            if len(all_concepts_data) < 2:
                print(f"[CROSS-AUTHOR] Not enough data for cross-author analysis (found {len(all_concepts_data)} concepts)")
                return None
            
            print(f"[CROSS-AUTHOR] Found {len(all_concepts_data)} concepts across {len(folder_names)} folders")
            
            # Create DataFrame
            df = pd.DataFrame(all_concepts_data)
            
            # 1. Generate Between-Author Similarity Heatmap
            print(f"[CROSS-AUTHOR] Generating heatmap...")
            between_author_sim = self._compute_between_author_similarity(df, folder_names)
            heatmap_path = self._create_between_author_heatmap(between_author_sim, folder_names, output_dir)
            print(f"[CROSS-AUTHOR] Heatmap result: {heatmap_path}")
            
            # 2. Generate 2D Embedding Map (t-SNE)
            print(f"[CROSS-AUTHOR] Generating t-SNE plot...")
            # Use the parameters passed to this function
            tsne_path = self._create_cross_author_tsne(df, folder_names, output_dir,
                                                       font_size=tsne_font_size,
                                                       n_components=tsne_n_components,
                                                       color_palette=tsne_color_palette,
                                                       proximity_threshold=tsne_proximity_threshold,
                                                       show_labels=tsne_show_labels)
            print(f"[CROSS-AUTHOR] t-SNE result: {tsne_path}")
            
            # Compute clustering once and share between both visualizations to ensure consistency
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            import numpy as np
            
            X = np.stack(df["embedding"])
            n_clusters = min(20, len(df) // 3)
            if n_clusters < 2:
                n_clusters = 2
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            cluster_centers = kmeans.cluster_centers_
            
            # Create shared PCA projection
            pca = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X)
            centers_2d = pca.transform(cluster_centers)
            
            # 2b. Generate alternative less-crowded visualization
            alt_viz_path = None
            try:
                print(f"[CROSS-AUTHOR] Generating alternative visualization...")
                alt_viz_path = self._create_alternative_clustering_viz(df, folder_names, output_dir, 
                                                                      cluster_labels, cluster_centers, X_2d, centers_2d)
                print(f"[CROSS-AUTHOR] Alternative visualization result: {alt_viz_path}")
            except Exception as alt_viz_error:
                print(f"[CROSS-AUTHOR] Failed to create alternative visualization: {alt_viz_error}")
                import traceback
                traceback.print_exc()
            
            # 2c. Generate varying radius clustering visualization
            radius_viz_path = None
            try:
                print(f"[CROSS-AUTHOR] Generating varying radius visualization...")
                radius_viz_path = self._create_varying_radius_clustering_viz(df, folder_names, output_dir,
                                                                             cluster_labels, cluster_centers, X_2d, centers_2d)
                print(f"[CROSS-AUTHOR] Varying radius visualization result: {radius_viz_path}")
            except Exception as radius_viz_error:
                print(f"[CROSS-AUTHOR] Failed to create varying radius visualization: {radius_viz_error}")
                import traceback
                traceback.print_exc()
            
            # 3. Generate Semantic Network Graph
            print(f"[CROSS-AUTHOR] Generating network graph...")
            network_path = self._create_semantic_network(df, folder_names, output_dir)
            print(f"[CROSS-AUTHOR] Network result: {network_path}")
            
            # 4. Generate CSV with results
            print(f"[CROSS-AUTHOR] Generating CSV...")
            csv_path = self._create_cross_author_csv(df, between_author_sim, folder_names, output_dir)
            print(f"[CROSS-AUTHOR] CSV result: {csv_path}")
            
            return {
                'heatmap': heatmap_path,
                'tsne': tsne_path,
                'alternative_viz': alt_viz_path,
                'varying_radius_viz': radius_viz_path,
                'network': network_path,
                'csv': csv_path,
                'between_author_sim': between_author_sim,
                'concepts_data': df
            }
            
        except Exception as e:
            print(f"[CROSS-AUTHOR ERROR] Failed to generate cross-author analysis: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _compute_between_author_similarity(self, df, folder_names):
        """Compute semantic similarity matrix between authors"""
        from sklearn.metrics.pairwise import cosine_similarity
        between_author_sim = np.zeros((len(folder_names), len(folder_names)))
        
        for i, author1 in enumerate(folder_names):
            for j, author2 in enumerate(folder_names):
                if i <= j:
                    # Get concepts for each author
                    author1_concepts = df[df['author'] == author1]['concept'].tolist()
                    author2_concepts = df[df['author'] == author2]['concept'].tolist()
                    
                    if author1_concepts and author2_concepts:
                        # Compute average similarity between all concept pairs
                        similarities = []
                        for c1 in author1_concepts:
                            for c2 in author2_concepts:
                                emb1 = df[(df['author'] == author1) & (df['concept'] == c1)]['embedding'].iloc[0]
                                emb2 = df[(df['author'] == author2) & (df['concept'] == c2)]['embedding'].iloc[0]
                                sim = cosine_similarity([emb1], [emb2])[0][0]
                                similarities.append(sim)
                        
                        avg_sim = np.mean(similarities) if similarities else 0.0
                        between_author_sim[i, j] = avg_sim
                        between_author_sim[j, i] = avg_sim  # Symmetric matrix
                    else:
                        between_author_sim[i, j] = 0.0
                        between_author_sim[j, i] = 0.0
        
        return between_author_sim

    def _create_between_author_heatmap(self, between_author_sim, folder_names, output_dir):
        """Create heatmap showing semantic similarity between authors"""
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(between_author_sim.astype(float), 
                       cmap="viridis", 
                       annot=True, 
                       xticklabels=folder_names,
                       yticklabels=folder_names,
                       fmt='.3f')
            plt.title("Semantic Similarity Between Authors\n(Darker color = higher semantic overlap)", 
                     fontsize=14, pad=20)
            plt.xlabel("Authors", fontsize=12)
            plt.ylabel("Authors", fontsize=12)
            plt.tight_layout()
            
            heatmap_path = os.path.join(output_dir, "cross_author_similarity_heatmap.png")
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[CROSS-AUTHOR] Created between-author heatmap: {heatmap_path}")
            return heatmap_path
        except Exception as e:
            print(f"[CROSS-AUTHOR ERROR] Failed to create heatmap: {e}")
            return None

    def _get_author_color_mapping(self, folder_names):
        """Create consistent color mapping for authors across all visualizations"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Use a fixed color palette and consistent ordering
        colors = plt.cm.Set3(np.linspace(0, 1, len(folder_names)))
        return {author: colors[i] for i, author in enumerate(sorted(folder_names))}

    def _create_cross_author_tsne(self, df, folder_names, output_dir, 
                                   font_size=5, n_components=2, color_palette='Set3', 
                                   proximity_threshold=None, show_labels=True):
        """Create 2D embedding map showing concept clustering by author
        
        Args:
            df: DataFrame with concepts and embeddings
            folder_names: List of folder/author names
            output_dir: Output directory for saving plots
            font_size: Font size for concept labels (default: 5)
            n_components: Number of dimensions for t-SNE (2 or 3, default: 2)
            color_palette: Matplotlib color palette name (default: 'Set3')
            proximity_threshold: If set, group nodes within this distance together (default: None)
            show_labels: Whether to show concept labels (default: True)
        """
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import numpy as np
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import pdist
            
            # Stack all embeddings
            X = np.stack(df["embedding"])
            
            # Apply t-SNE with configurable dimensions
            n_components = max(2, min(3, int(n_components)))  # Ensure 2 or 3
            tsne = TSNE(n_components=n_components, perplexity=min(20, len(df)-1), random_state=42)
            X_2d = tsne.fit_transform(X)
            
            # Get color palette
            try:
                cmap = plt.cm.get_cmap(color_palette)
                colors = cmap(np.linspace(0, 1, len(folder_names)))
                author_colors = {author: colors[i] for i, author in enumerate(sorted(folder_names))}
            except:
                # Fallback to default
                author_colors = self._get_author_color_mapping(folder_names)
            
            # Apply proximity grouping if threshold is set
            cluster_labels = None
            if proximity_threshold is not None and proximity_threshold > 0:
                try:
                    # Compute pairwise distances
                    distances = pdist(X_2d)
                    # Perform hierarchical clustering
                    linkage_matrix = linkage(distances, method='ward')
                    # Create clusters based on threshold
                    cluster_labels = fcluster(linkage_matrix, proximity_threshold, criterion='distance')
                except:
                    cluster_labels = None
            
            # Create plot (2D or 3D)
            if n_components == 3:
                fig = plt.figure(figsize=(14, 12))
                ax = fig.add_subplot(111, projection='3d')
            else:
                fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot by author
            for author in folder_names:
                mask = df["author"] == author
                if mask.any():
                    author_data = df[mask]
                    concept_count = len(author_data)
                    author_indices = df[mask].index
                    
                    # Get coordinates for this author
                    if n_components == 3:
                        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], X_2d[mask, 2],
                                  label=f"{author} ({concept_count} concepts)", 
                                  alpha=0.7, s=100, c=[author_colors[author]])
                    else:
                        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                                  label=f"{author} ({concept_count} concepts)", 
                                  alpha=0.7, s=100, c=[author_colors[author]])
                    
                    # Add concept labels if enabled
                    if show_labels:
                        # Show fewer labels if there are many concepts
                        max_labels = min(int(concept_count * 0.9), concept_count) if concept_count > 50 else concept_count
                        
                        for idx, (_, row) in enumerate(author_data.iterrows()):
                            if idx < max_labels:
                                concept = row['concept']
                                # Truncate long concept names
                                display_concept = concept[:12] + '...' if len(concept) > 12 else concept
                                if n_components == 3:
                                    ax.text(X_2d[mask][idx, 0], X_2d[mask][idx, 1], X_2d[mask][idx, 2],
                                           display_concept, fontsize=font_size, alpha=0.8)
                                else:
                                    ax.annotate(display_concept, 
                                               (X_2d[mask][idx, 0], X_2d[mask][idx, 1]),
                                               xytext=(2, 2), textcoords='offset points',
                                               fontsize=font_size, alpha=0.8)
            
            # Add cluster grouping visualization if enabled
            if cluster_labels is not None:
                # Draw circles/ellipses around clusters
                unique_clusters = np.unique(cluster_labels)
                for cluster_id in unique_clusters:
                    cluster_mask = cluster_labels == cluster_id
                    if np.sum(cluster_mask) > 1:  # Only draw for clusters with multiple points
                        cluster_points = X_2d[cluster_mask]
                        if n_components == 3:
                            # For 3D, draw a sphere approximation
                            center = cluster_points.mean(axis=0)
                            radius = np.max(np.linalg.norm(cluster_points - center, axis=1))
                            # Draw a simple circle in the XY plane
                            theta = np.linspace(0, 2*np.pi, 100)
                            ax.plot(center[0] + radius * np.cos(theta), 
                                   center[1] + radius * np.sin(theta),
                                   center[2], 'k--', alpha=0.3, linewidth=1)
                        else:
                            # For 2D, draw a circle
                            center = cluster_points.mean(axis=0)
                            radius = np.max(np.linalg.norm(cluster_points - center, axis=1))
                            circle = plt.Circle(center, radius, fill=False, linestyle='--', 
                                              alpha=0.3, color='gray', linewidth=1)
                            ax.add_patch(circle)
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_title("Concept Embeddings by Author\n(Closer points = more similar concepts)", 
                     fontsize=14, pad=20)
            if n_components == 3:
                ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
                ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
                ax.set_zlabel("t-SNE Dimension 3", fontsize=12)
            else:
                ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
                ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            tsne_path = os.path.join(output_dir, "cross_author_tsne.png")
            plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[CROSS-AUTHOR] Created cross-author t-SNE: {tsne_path}")
            return tsne_path
        except Exception as e:
            print(f"[CROSS-AUTHOR ERROR] Failed to create t-SNE: {e}")
            return None
    
    def _create_alternative_clustering_viz(self, df, folder_names, output_dir, 
                                          cluster_labels=None, cluster_centers=None, X_2d=None, centers_2d=None):
        """Create an alternative, less crowded visualization using cluster centers with pie charts showing composition
        
        Args:
            df: DataFrame with concepts and embeddings
            folder_names: List of folder/author names
            output_dir: Output directory
            cluster_labels: Pre-computed cluster labels (optional, will compute if None)
            cluster_centers: Pre-computed cluster centers (optional)
            X_2d: Pre-computed 2D projection of embeddings (optional)
            centers_2d: Pre-computed 2D projection of cluster centers (optional)
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
            import numpy as np
            from collections import Counter
            
            # Use provided clustering or compute new
            if cluster_labels is None:
                from sklearn.cluster import KMeans
                from sklearn.decomposition import PCA
                X = np.stack(df["embedding"])
                n_clusters = min(20, len(df) // 3)
                if n_clusters < 2:
                    n_clusters = 2
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X)
                cluster_centers = kmeans.cluster_centers_
                pca = PCA(n_components=2, random_state=42)
                X_2d = pca.fit_transform(X)
                centers_2d = pca.transform(cluster_centers)
            
            n_clusters = len(np.unique(cluster_labels))
            
            # For each cluster, find the most representative concept (closest to center)
            representative_concepts = []
            # Need original embeddings for distance calculation
            X_full = np.stack(df["embedding"])
            
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                if np.sum(cluster_mask) > 0:
                    cluster_points = X_full[cluster_mask]
                    cluster_center = cluster_centers[cluster_id]
                    # Find closest point to center
                    distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
                    closest_idx = np.argmin(distances)
                    cluster_df_indices = df.index[cluster_mask]
                    representative_idx = cluster_df_indices[closest_idx]
                    representative_concepts.append(representative_idx)
            
            # Calculate minimum distance between cluster centers to prevent overlap
            min_distance = float('inf')
            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    dist = np.linalg.norm(centers_2d[i] - centers_2d[j])
                    if dist < min_distance:
                        min_distance = dist
            
            # Calculate plot extent to determine appropriate pie chart size
            x_range = np.max(centers_2d[:, 0]) - np.min(centers_2d[:, 0])
            y_range = np.max(centers_2d[:, 1]) - np.min(centers_2d[:, 1])
            plot_extent = max(x_range, y_range)
            
            # Set pie chart radius with minimum size for visibility
            # Use 6-10% of plot extent as base, with minimum size guarantee
            min_radius_absolute = plot_extent * 0.06  # Minimum 6% of plot extent - ensures visibility
            base_radius_relative = max(min_distance * 0.25, min_radius_absolute)  # At least minimum size
            max_radius_relative = min(min_distance * 0.4, plot_extent * 0.10)  # Maximum 10%
            min_radius_relative = min_radius_absolute  # Use minimum for all
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(16, 14))
            
            # Get color mapping
            author_colors = self._get_author_color_mapping(folder_names)
            
            # Plot cluster members first (so they appear behind the centers)
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                if np.sum(cluster_mask) > 0:
                    cluster_points_2d = X_2d[cluster_mask]
                    cluster_df_subset = df[cluster_mask]
                    
                    # Plot each member with its actual author color - make them larger and more visible
                    for idx, (_, row) in enumerate(cluster_df_subset.iterrows()):
                        author = row['author']
                        ax.scatter(cluster_points_2d[idx, 0], cluster_points_2d[idx, 1],
                                  s=80, c=[author_colors[author]], alpha=0.6, 
                                  edgecolors='white', linewidths=0.5, marker='o')
                    
                    # Draw dotted lines from cluster center to members (after center is drawn)
                    # This will be done after the pie chart is drawn
            
            # Plot cluster centers with pie charts showing composition
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                if np.sum(cluster_mask) > 0:
                    cluster_df_subset = df[cluster_mask]
                    
                    # Count authors in this cluster
                    author_counts = Counter(cluster_df_subset['author'])
                    total_count = sum(author_counts.values())
                    
                    # Get the representative concept
                    rep_idx = representative_concepts[cluster_id]
                    concept = df.loc[rep_idx, 'concept']
                    
                    center_x, center_y = centers_2d[cluster_id, 0], centers_2d[cluster_id, 1]
                    
                    # Draw pie chart showing author composition
                    # Use minimum size to ensure visibility, scale slightly with cluster size
                    size_factor = min(total_count / 20, 1.0)  # Scale from 0 to 1 based on cluster size
                    pie_radius = min_radius_relative + (base_radius_relative - min_radius_relative) * size_factor
                    pie_radius = min(pie_radius, max_radius_relative)  # Cap at maximum
                    
                    # Check for overlap and adjust center position if needed (conservative approach)
                    adjusted_x, adjusted_y = center_x, center_y
                    max_adjustment = pie_radius * 0.3  # Don't move more than 30% of radius
                    
                    for other_cluster_id in range(n_clusters):
                        if other_cluster_id != cluster_id:
                            other_mask = cluster_labels == other_cluster_id
                            if np.sum(other_mask) > 0:
                                other_center = centers_2d[other_cluster_id]
                                other_count = sum(Counter(df[other_mask]['author']).values())
                                # Calculate other cluster's radius
                                other_size_factor = min(other_count / 20, 1.0)
                                other_radius = min_radius_relative + (base_radius_relative - min_radius_relative) * other_size_factor
                                other_radius = min(other_radius, max_radius_relative)
                                
                                # Check distance
                                dist = np.linalg.norm([adjusted_x - other_center[0], adjusted_y - other_center[1]])
                                min_required_dist = (pie_radius + other_radius) * 1.1  # 10% margin
                                
                                if dist < min_required_dist:
                                    # Move center slightly away (conservative)
                                    direction = np.array([adjusted_x - other_center[0], adjusted_y - other_center[1]])
                                    if np.linalg.norm(direction) > 0:
                                        direction = direction / np.linalg.norm(direction)
                                        move_distance = min((min_required_dist - dist) * 0.3, max_adjustment)
                                        adjusted_x = adjusted_x + direction[0] * move_distance
                                        adjusted_y = adjusted_y + direction[1] * move_distance
                    
                    center_x, center_y = adjusted_x, adjusted_y
                    
                    if len(author_counts) > 1:
                        # Multiple authors - create pie chart
                        sizes = [author_counts[author] for author in sorted(author_counts.keys())]
                        colors_list = [author_colors[author] for author in sorted(author_counts.keys())]
                        
                        # Create pie chart using wedges
                        if sizes and sum(sizes) > 0:
                            start_angle = 90  # Start at top
                            for size, color in zip(sizes, colors_list):
                                angle = 360 * (size / total_count)
                                # Draw wedge
                                theta = np.linspace(0, angle, 50)
                                x_wedge = center_x + pie_radius * np.cos(np.radians(theta + start_angle))
                                y_wedge = center_y + pie_radius * np.sin(np.radians(theta + start_angle))
                                ax.fill([center_x] + list(x_wedge) + [center_x], 
                                       [center_y] + list(y_wedge) + [center_y],
                                       color=color, alpha=0.6, edgecolor='black', linewidth=1.2)
                                start_angle += angle
                    else:
                        # Single author cluster - draw a solid circle
                        single_author = list(author_counts.keys())[0]
                        circle = Circle((center_x, center_y), pie_radius, 
                                       color=author_colors[single_author], 
                                       alpha=0.6, edgecolor='black', linewidth=2)
                        ax.add_patch(circle)
                    
                    # Add label for representative concept (shortened, no parentheses text)
                    # Remove any text in parentheses if present
                    concept_label = concept.split('(')[0].strip() if '(' in concept else concept
                    concept_label = concept_label[:20] + ('...' if len(concept_label) > 20 else '')
                    ax.annotate(concept_label,
                               (center_x, center_y),
                               xytext=(8, 8), textcoords='offset points',
                               fontsize=10, alpha=0.7,  # More transparent
                               bbox=dict(boxstyle='round,pad=0.5',
                               facecolor='white', alpha=0.6, edgecolor='black', linewidth=1),  # More transparent
                               ha='left')
                    
                    # Add cluster size annotation (more transparent)
                    ax.annotate(f"n={total_count}",
                               (center_x, center_y),
                               xytext=(8, -15), textcoords='offset points',
                               fontsize=8, alpha=0.6,  # More transparent
                               bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='lightgray', alpha=0.5),  # More transparent
                               ha='left')
                    
                    # Draw dotted lines from pie chart center to cluster members
                    cluster_points_2d = X_2d[cluster_mask]
                    for point in cluster_points_2d:
                        ax.plot([center_x, point[0]], [center_y, point[1]],
                               'k--', alpha=0.2, linewidth=0.8, zorder=0)  # Dotted lines, very transparent
                    
                    # Draw a light shaded area (convex hull) around cluster members
                    if len(cluster_points_2d) > 2:
                        from scipy.spatial import ConvexHull
                        try:
                            hull = ConvexHull(cluster_points_2d)
                            hull_points = cluster_points_2d[hull.vertices]
                            # Add center point to create a more connected area
                            hull_with_center = np.vstack([cluster_points_2d[hull.vertices], [center_x, center_y]])
                            hull_extended = ConvexHull(hull_with_center)
                            ax.fill(hull_with_center[hull_extended.vertices, 0],
                                   hull_with_center[hull_extended.vertices, 1],
                                   alpha=0.1, color='gray', edgecolor='none', zorder=0)
                        except:
                            # If convex hull fails, just skip it
                            pass
            
            # Add legend for authors
            legend_elements = []
            for author in folder_names:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                 markerfacecolor=author_colors[author],
                                                 markersize=10, label=author, alpha=0.7))
            
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', title='Authors')
            ax.set_title("Concept Clusters (Less Crowded View)\nPie charts show author composition, points show individual concepts", 
                        fontsize=14, pad=20)
            ax.set_xlabel("PCA Dimension 1", fontsize=12)
            ax.set_ylabel("PCA Dimension 2", fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            alt_viz_path = os.path.join(output_dir, "cross_author_alternative_clustering.png")
            plt.savefig(alt_viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[CROSS-AUTHOR] Created alternative clustering visualization: {alt_viz_path}")
            return alt_viz_path
        except Exception as e:
            print(f"[CROSS-AUTHOR ERROR] Failed to create alternative visualization: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_varying_radius_clustering_viz(self, df, folder_names, output_dir,
                                             cluster_labels=None, cluster_centers=None, X_2d=None, centers_2d=None):
        """Create a visualization with clusters shown as circles with varying radius based on member count
        
        Args:
            df: DataFrame with concepts and embeddings
            folder_names: List of folder/author names
            output_dir: Output directory
            cluster_labels: Pre-computed cluster labels (optional, will compute if None)
            cluster_centers: Pre-computed cluster centers (optional)
            X_2d: Pre-computed 2D projection of embeddings (optional)
            centers_2d: Pre-computed 2D projection of cluster centers (optional)
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
            import numpy as np
            from collections import Counter
            
            # Use provided clustering or compute new
            if cluster_labels is None:
                from sklearn.cluster import KMeans
                from sklearn.decomposition import PCA
                X = np.stack(df["embedding"])
                n_clusters = min(20, len(df) // 3)
                if n_clusters < 2:
                    n_clusters = 2
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X)
                cluster_centers = kmeans.cluster_centers_
                pca = PCA(n_components=2, random_state=42)
                X_2d = pca.fit_transform(X)
                centers_2d = pca.transform(cluster_centers)
            
            n_clusters = len(np.unique(cluster_labels))
            
            # Calculate cluster sizes and prepare data
            cluster_data = []
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                if np.sum(cluster_mask) > 0:
                    cluster_df_subset = df[cluster_mask]
                    author_counts = Counter(cluster_df_subset['author'])
                    total_count = sum(author_counts.values())
                    center_x, center_y = centers_2d[cluster_id, 0], centers_2d[cluster_id, 1]
                    cluster_data.append({
                        'id': cluster_id,
                        'center': (center_x, center_y),
                        'count': total_count,
                        'author_counts': author_counts,
                        'points': X_2d[cluster_mask],
                        'df_subset': cluster_df_subset
                    })
            
            # Calculate minimum distance between cluster centers
            min_distance = float('inf')
            for i, cluster_i in enumerate(cluster_data):
                for j, cluster_j in enumerate(cluster_data):
                    if i < j:
                        dist = np.linalg.norm(np.array(cluster_i['center']) - np.array(cluster_j['center']))
                        if dist < min_distance:
                            min_distance = dist
            
            # Calculate plot extent
            x_coords = [c['center'][0] for c in cluster_data]
            y_coords = [c['center'][1] for c in cluster_data]
            x_range = max(x_coords) - min(x_coords) if x_coords else 1.0
            y_range = max(y_coords) - min(y_coords) if y_coords else 1.0
            plot_extent = max(x_range, y_range)
            
            # Calculate radius scale based on cluster counts
            max_count = max([c['count'] for c in cluster_data]) if cluster_data else 1
            min_count = min([c['count'] for c in cluster_data]) if cluster_data else 1
            
            # Set base radius to ensure no overlap
            # Use 20-40% of minimum distance, scaled by cluster size
            base_radius = min(min_distance * 0.2, plot_extent * 0.04)
            max_radius = min(min_distance * 0.4, plot_extent * 0.08)
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(16, 14))
            
            # Get color mapping
            author_colors = self._get_author_color_mapping(folder_names)
            
            # Find representative concepts for each cluster (for labels)
            X_full = np.stack(df["embedding"])
            representative_concepts = {}
            for cluster_info in cluster_data:
                cluster_id = cluster_info['id']
                cluster_mask = cluster_labels == cluster_id
                if np.sum(cluster_mask) > 0:
                    cluster_points = X_full[cluster_mask]
                    cluster_center = cluster_centers[cluster_id]
                    # Find closest point to center
                    distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
                    closest_idx = np.argmin(distances)
                    cluster_df_indices = df.index[cluster_mask]
                    representative_idx = cluster_df_indices[closest_idx]
                    representative_concepts[cluster_id] = df.loc[representative_idx, 'concept']
            
            # Plot cluster members first
            for cluster_info in cluster_data:
                for idx, (_, row) in enumerate(cluster_info['df_subset'].iterrows()):
                    author = row['author']
                    point = cluster_info['points'][idx]
                    ax.scatter(point[0], point[1],
                              s=60, c=[author_colors[author]], alpha=0.5, 
                              edgecolors='white', linewidths=0.3, marker='o', zorder=1)
            
            # Plot clusters as circles with varying radius
            for cluster_info in cluster_data:
                center_x, center_y = cluster_info['center']
                count = cluster_info['count']
                author_counts = cluster_info['author_counts']
                
                # Calculate radius based on cluster size
                # Scale from base_radius to max_radius based on count
                if max_count > min_count:
                    size_factor = (count - min_count) / (max_count - min_count)
                else:
                    size_factor = 0.5
                radius = base_radius + (max_radius - base_radius) * size_factor
                
                # Check for overlap with other clusters and adjust if needed
                for other_cluster in cluster_data:
                    if other_cluster['id'] != cluster_info['id']:
                        other_center = other_cluster['center']
                        other_count = other_cluster['count']
                        dist = np.linalg.norm(np.array([center_x, center_y]) - np.array(other_center))
                        
                        # Calculate other cluster's radius
                        if max_count > min_count:
                            other_size_factor = (other_count - min_count) / (max_count - min_count)
                        else:
                            other_size_factor = 0.5
                        other_radius = base_radius + (max_radius - base_radius) * other_size_factor
                        
                        # If circles would overlap, reduce radius
                        if dist < (radius + other_radius) * 1.1:  # 10% margin
                            max_allowed_radius = dist / 2.2  # Leave 10% gap
                            radius = min(radius, max_allowed_radius)
                
                # Determine circle color - use dominant author or mixed color
                if len(author_counts) == 1:
                    # Single author - use that color
                    dominant_author = list(author_counts.keys())[0]
                    circle_color = author_colors[dominant_author]
                else:
                    # Multiple authors - use a weighted average color
                    total = sum(author_counts.values())
                    color_sum = np.array([0.0, 0.0, 0.0, 1.0])  # RGBA
                    for author, count in author_counts.items():
                        # Convert color to RGBA
                        if isinstance(author_colors[author], str):
                            from matplotlib.colors import to_rgba
                            color_rgba = np.array(to_rgba(author_colors[author]))
                        else:
                            # Already a tuple/array
                            color_rgba = np.array(author_colors[author])
                            if len(color_rgba) == 3:
                                color_rgba = np.append(color_rgba, 1.0)  # Add alpha
                        color_sum[:3] += color_rgba[:3] * (count / total)
                    circle_color = tuple(color_sum)
                
                # Draw circle
                circle = Circle((center_x, center_y), radius,
                               color=circle_color, alpha=0.6,
                               edgecolor='black', linewidth=2, zorder=2)
                ax.add_patch(circle)
                
                # Add count label
                ax.annotate(f"n={count}",
                           (center_x, center_y),
                           ha='center', va='center',
                           fontsize=9, fontweight='bold',
                           color='white' if np.mean(circle_color[:3]) < 0.5 else 'black',
                           zorder=3)
                
                # Add representative concept label (similar to Alternative Clustering)
                if cluster_info['id'] in representative_concepts:
                    concept = representative_concepts[cluster_info['id']]
                    # Remove any text in parentheses if present
                    concept_label = concept.split('(')[0].strip() if '(' in concept else concept
                    concept_label = concept_label[:20] + ('...' if len(concept_label) > 20 else '')
                    ax.annotate(concept_label,
                               (center_x, center_y),
                               xytext=(8, -25), textcoords='offset points',
                               fontsize=9, alpha=0.7,
                               bbox=dict(boxstyle='round,pad=0.4',
                               facecolor='white', alpha=0.6, edgecolor='black', linewidth=1),
                               ha='left', zorder=4)
                
                # Draw dotted lines from circle center to cluster members
                for point in cluster_info['points']:
                    ax.plot([center_x, point[0]], [center_y, point[1]],
                           'k--', alpha=0.2, linewidth=0.8, zorder=0)  # Dotted lines, very transparent
                
                # Draw a light shaded area (convex hull) around cluster members
                if len(cluster_info['points']) > 2:
                    from scipy.spatial import ConvexHull
                    try:
                        hull = ConvexHull(cluster_info['points'])
                        hull_points = cluster_info['points'][hull.vertices]
                        # Add center point to create a more connected area
                        hull_with_center = np.vstack([cluster_info['points'][hull.vertices], [center_x, center_y]])
                        hull_extended = ConvexHull(hull_with_center)
                        ax.fill(hull_with_center[hull_extended.vertices, 0],
                               hull_with_center[hull_extended.vertices, 1],
                               alpha=0.1, color='gray', edgecolor='none', zorder=0)
                    except:
                        # If convex hull fails, just skip it
                        pass
            
            # Add legend for authors
            legend_elements = []
            for author in folder_names:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                 markerfacecolor=author_colors[author],
                                                 markersize=10, label=author, alpha=0.7))
            
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', title='Authors')
            ax.set_title("Concept Clusters with Varying Radius\nCircle size represents cluster member count", 
                        fontsize=14, pad=20)
            ax.set_xlabel("PCA Dimension 1", fontsize=12)
            ax.set_ylabel("PCA Dimension 2", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            plt.tight_layout()
            
            radius_viz_path = os.path.join(output_dir, "cross_author_varying_radius_clustering.png")
            plt.savefig(radius_viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[CROSS-AUTHOR] Created varying radius clustering visualization: {radius_viz_path}")
            return radius_viz_path
        except Exception as e:
            print(f"[CROSS-AUTHOR ERROR] Failed to create varying radius visualization: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_semantic_network(self, df, folder_names, output_dir):
        """Create semantic network graph showing concept relationships"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            import networkx as nx
            import matplotlib.pyplot as plt
            import numpy as np
            
            G = nx.Graph()
            threshold = 0.8  # Similarity threshold for edges
            
            # Add nodes (concepts) with author information
            for _, row in df.iterrows():
                G.add_node(row['concept'], author=row['author'])
            
            # Add edges based on similarity
            for i, row_i in df.iterrows():
                for j, row_j in df.iterrows():
                    if i < j:
                        sim = cosine_similarity([row_i['embedding']], [row_j['embedding']])[0][0]
                        if sim > threshold:
                            G.add_edge(row_i['concept'], row_j['concept'], weight=sim)
            
            # Create the plot
            plt.figure(figsize=(15, 12))
            
            # Position nodes using spring layout
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Color nodes by author using consistent mapping
            author_colors = self._get_author_color_mapping(folder_names)
            node_colors = [author_colors[G.nodes[node]['author']] for node in G.nodes()]
            
            # Draw the network
            nx.draw(G, pos, 
                   node_color=node_colors,
                   node_size=100,
                   with_labels=False,
                   alpha=0.7,
                   edge_color='gray',
                   width=0.5)
            
            # Add concept labels for all nodes (or 90% if too many)
            total_nodes = len(G.nodes())
            max_labels = min(int(total_nodes * 0.9), total_nodes) if total_nodes > 50 else total_nodes
            
            if max_labels > 0:
                # Select nodes to label (first max_labels nodes)
                nodes_to_label = list(G.nodes())[:max_labels]
                label_pos = {node: pos[node] for node in nodes_to_label}
                
                # Truncate long concept names for display
                labels = {node: node[:8] + '...' if len(node) > 8 else node 
                         for node in nodes_to_label}
                nx.draw_networkx_labels(G, label_pos, labels, font_size=8, alpha=0.8)
            
            # Add legend for authors with concept counts
            legend_elements = []
            for author in folder_names:
                author_nodes = [node for node in G.nodes() if G.nodes[node]['author'] == author]
                concept_count = len(author_nodes)
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=author_colors[author], 
                                                markersize=10, label=f"{author} ({concept_count} concepts)"))
            plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
            
            plt.title(f"Semantic Concept Network\n(Edges: similarity > {threshold})", 
                     fontsize=14, pad=20)
            plt.tight_layout()
            
            network_path = os.path.join(output_dir, "cross_author_network.png")
            plt.savefig(network_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[CROSS-AUTHOR] Created semantic network: {network_path}")
            return network_path
        except Exception as e:
            print(f"[CROSS-AUTHOR ERROR] Failed to create network: {e}")
            return None

    def _create_cross_author_csv(self, df, between_author_sim, folder_names, output_dir):
        """Create CSV file with cross-author analysis results"""
        try:
            # Create similarity matrix DataFrame
            sim_df = pd.DataFrame(between_author_sim, 
                                index=folder_names, 
                                columns=folder_names)
            
            # Create concepts summary with ALL concepts for each author
            concepts_summary = []
            print(f"[CROSS-AUTHOR] Creating concepts summary for {len(folder_names)} authors")
            print(f"[CROSS-AUTHOR] Folder names: {folder_names}")
            print(f"[CROSS-AUTHOR] DataFrame shape: {df.shape}")
            print(f"[CROSS-AUTHOR] DataFrame columns: {df.columns.tolist()}")
            print(f"[CROSS-AUTHOR] Unique authors in DataFrame: {df['author'].unique().tolist()}")
            
            for author in folder_names:
                author_concepts = df[df['author'] == author]['concept'].tolist()
                print(f"[CROSS-AUTHOR] Author '{author}' has {len(author_concepts)} concepts")
                if len(author_concepts) <= 5:
                    print(f"[CROSS-AUTHOR] Concepts for '{author}': {author_concepts}")
                
                concepts_summary.append({
                    'author': author,
                    'concept_count': len(author_concepts),
                    'all_concepts': ' | '.join(author_concepts),  # Use pipe separator instead of semicolon to avoid CSV issues
                    'concepts_sample': ' | '.join(author_concepts[:10]) + ('...' if len(author_concepts) > 10 else ''),
                    'embedding_model': self.model_name if hasattr(self, 'model_name') else 'sentence-transformers/all-MiniLM-L6-v2'
                })
            
            concepts_df = pd.DataFrame(concepts_summary)
            
            # Save to CSV file (main similarity matrix)
            csv_path = os.path.join(output_dir, "cross_author_analysis.csv")
            sim_df.to_csv(csv_path, index=True)
            
            # Also save concepts summary as separate CSV
            concepts_csv_path = os.path.join(output_dir, "cross_author_concepts_summary.csv")
            print(f"[CROSS-AUTHOR] Saving concepts summary to: {concepts_csv_path}")
            print(f"[CROSS-AUTHOR] Concepts DataFrame shape: {concepts_df.shape}")
            print(f"[CROSS-AUTHOR] Concepts DataFrame columns: {concepts_df.columns.tolist()}")
            print(f"[CROSS-AUTHOR] Concepts DataFrame preview:")
            print(concepts_df.head())
            # Use proper CSV escaping to handle commas in concept names
            concepts_df.to_csv(concepts_csv_path, index=False, quoting=1)  # quoting=1 means quote all fields
            print(f"[CROSS-AUTHOR] Successfully saved concepts summary CSV")
            
            # Also save individual CSV files
            csv_sim_path = os.path.join(output_dir, "cross_author_similarity_matrix.csv")
            # Note: concepts_csv_path is already defined above, no need to redefine
            
            sim_df.to_csv(csv_sim_path, index=True)
            # Note: concepts_df is already saved above, no need to save again
            
            # Return the CSV file as the main result
            # csv_path is already defined above
            
            print(f"[CROSS-AUTHOR] Created CSV: {csv_path}")
            return csv_path
        except Exception as e:
            print(f"[CROSS-AUTHOR ERROR] Failed to create CSV: {e}")
            return None

    def generate_analysis_csv(self, analysis_type, results, folder_name, output_dir):
        """Generate CSV files for different analysis types (A, B, C)"""
        try:
            import os
            import pandas as pd
            
            csv_data = []
            embedding_model = getattr(self, 'model_name', 'sentence-transformers/all-MiniLM-L6-v2')
            
            if analysis_type == 'within_param' and 'individual_parameters' in results:
                # A. Within-Parameter Analysis CSV
                for param, param_results in results['individual_parameters'].items():
                    csv_data.append({
                        'analysis_type': 'Within-Parameter Analysis',
                        'parameter': param,
                        'semantic_mean': param_results.get('overall_semantic_mean', 0.0),
                        'semantic_std': param_results.get('overall_semantic_std', 0.0),
                        'exact_mean': param_results.get('overall_exact_mean', 0.0),
                        'exact_std': param_results.get('overall_exact_std', 0.0),
                        'total_comparisons': param_results.get('total_comparisons', 0),
                        'folder_name': folder_name,
                        'embedding_model': embedding_model
                    })
                    
            elif analysis_type == 'cross_param' and 'cross_parameter' in results:
                # B. Cross-Parameter Analysis CSV
                cross_results = results['cross_parameter']
                csv_data.append({
                    'analysis_type': 'Cross-Parameter Analysis',
                    'parameter': 'All Parameters',
                    'semantic_mean': cross_results.get('overall_semantic_mean', 0.0),
                    'semantic_std': cross_results.get('overall_semantic_std', 0.0),
                    'exact_mean': cross_results.get('overall_exact_mean', 0.0),
                    'exact_std': cross_results.get('overall_exact_std', 0.0),
                    'total_comparisons': cross_results.get('total_comparisons', 0),
                    'folder_name': folder_name,
                    'embedding_model': embedding_model
                })
                
            elif analysis_type == 'sensitivity' and 'individual_parameters' in results:
                # C. Parameter Sensitivity Ranking CSV
                sensitivity_scores = []
                for param, param_results in results['individual_parameters'].items():
                    semantic_mean = param_results.get('overall_semantic_mean', 0.0)
                    exact_mean = param_results.get('overall_exact_mean', 0.0)
                    # Lower values = higher sensitivity
                    sensitivity_score = 1.0 - ((semantic_mean + exact_mean) / 2.0)
                    sensitivity_scores.append((param, sensitivity_score, semantic_mean, exact_mean))
                
                # Sort by sensitivity (highest first)
                sensitivity_scores.sort(key=lambda x: x[1], reverse=True)
                
                for rank, (param, sensitivity, semantic, exact) in enumerate(sensitivity_scores, 1):
                    csv_data.append({
                        'analysis_type': 'Parameter Sensitivity Ranking',
                        'parameter': param,
                        'sensitivity_rank': rank,
                        'sensitivity_score': sensitivity,
                        'semantic_mean': semantic,
                        'exact_mean': exact,
                        'folder_name': folder_name,
                        'embedding_model': embedding_model
                    })
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_path = os.path.join(output_dir, f"{folder_name}_{analysis_type}_analysis.csv")
                df.to_csv(csv_path, index=False)
                print(f"[CSV EXPORT] Created {analysis_type} CSV: {csv_path}")
                return csv_path
            
            return None
            
        except Exception as e:
            print(f"[CSV EXPORT ERROR] Failed to create {analysis_type} CSV: {e}")
            return None

def create_fixed_width_table(doc, rows, cols, col_widths_inches):
    """
    Create a table with fixed column widths that actually work in python-docx.
    
    Args:
        doc: The docx Document object
        rows: Number of rows
        cols: Number of columns
        col_widths_inches: List of column widths in inches [3, 9]
    
    Returns:
        The created table object
    """
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches
    
    # Create the table
    table = doc.add_table(rows=rows, cols=cols)
    
    # Set table-level properties first
    table.style = 'Table Grid'
    table.autofit = False
    
    # Get table element
    tbl = table._tbl
    
    # Remove any existing table properties and start fresh
    tblPr = tbl.tblPr
    if tblPr is None:
        tblPr = OxmlElement('w:tblPr')
        tbl.insert(0, tblPr)
    
    # Clear existing table width if any
    for child in list(tblPr):
        if child.tag.endswith('tblW'):
            tblPr.remove(child)
        if child.tag.endswith('tblLayout'):
            tblPr.remove(child)
    
    # Set table layout to fixed (CRITICAL)
    tblLayout = OxmlElement('w:tblLayout')
    tblLayout.set(qn('w:type'), 'fixed')
    tblPr.append(tblLayout)
    
    # Set total table width
    total_width_twips = sum(int(w * 1440) for w in col_widths_inches)
    tblW = OxmlElement('w:tblW')
    tblW.set(qn('w:w'), str(total_width_twips))
    tblW.set(qn('w:type'), 'dxa')
    tblPr.append(tblW)
    
    # Set column widths using grid (CRITICAL PART)
    tblGrid = tbl.tblGrid
    if tblGrid is not None:
        tbl.remove(tblGrid)
    
    # Create new grid with specific widths
    tblGrid = OxmlElement('w:tblGrid')
    for width_inches in col_widths_inches:
        width_twips = int(width_inches * 1440)  # Convert inches to twips
        gridCol = OxmlElement('w:gridCol')
        gridCol.set(qn('w:w'), str(width_twips))
        tblGrid.append(gridCol)
    tbl.append(tblGrid)
    
    # MOST IMPORTANT: Set cell widths for all cells - FORCE the widths
    for row_idx, row in enumerate(table.rows):
        for cell_idx, cell in enumerate(row.cells):
            if cell_idx < len(col_widths_inches):
                width_twips = int(col_widths_inches[cell_idx] * 1440)
                
                # Get the cell's XML element
                tc = cell._tc
                
                # Remove existing tcPr and create fresh one
                tcPr_elements = [child for child in tc if child.tag.endswith('tcPr')]
                for tcPr in tcPr_elements:
                    tc.remove(tcPr)
                
                # Create completely new tcPr
                tcPr = OxmlElement('w:tcPr')
                
                # Set cell width with highest priority
                tcW = OxmlElement('w:tcW')
                tcW.set(qn('w:w'), str(width_twips))
                tcW.set(qn('w:type'), 'dxa')
                tcPr.append(tcW)
                
                # Add table cell margins to prevent overflow
                tcMar = OxmlElement('w:tcMar')
                for side in ['left', 'right', 'top', 'bottom']:
                    mar = OxmlElement(f'w:{side}')
                    mar.set(qn('w:w'), '100')
                    mar.set(qn('w:type'), 'dxa')
                    tcMar.append(mar)
                tcPr.append(tcMar)
                
                # Insert tcPr as first child
                tc.insert(0, tcPr)
                
                # Force the width property
                cell.width = Inches(col_widths_inches[cell_idx])
                
                print(f"Set cell {cell_idx} width to {width_twips} twips ({col_widths_inches[cell_idx]} inches)")
    
    return table
    
def wrap_label(text, width=18):
    # Try to wrap at word boundaries, fallback to hard wrap
    return '\n'.join(textwrap.wrap(text, width=width))

class ZoomablePanCanvas:
    def __init__(self, parent):
        self.parent = parent
        self.canvas = tk.Canvas(parent, bg='white')
        
        # Scrollbars
        self.h_scrollbar = ttk.Scrollbar(parent, orient="horizontal", command=self.canvas.xview)
        self.v_scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.h_scrollbar.set, yscrollcommand=self.v_scrollbar.set)
        
        # Grid layout
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure grid weights
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        
        # Zoom and pan variables
        self.zoom_level = 1.0
        self.original_images = []
        self.image_refs = []
        self.image_items = []
        
        # Bind mouse events for panning
        self.canvas.bind("<Button-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.do_pan)
        self.canvas.bind("<MouseWheel>", self.zoom_wheel)
        
        self.last_x = 0
        self.last_y = 0
        
    def start_pan(self, event):
        self.canvas.scan_mark(event.x, event.y)
        self.last_x = event.x
        self.last_y = event.y
        
    def do_pan(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        
    def zoom_wheel(self, event):
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()
            
    def set_images(self, image_paths, window_width):
        self.original_images = []
        self.image_refs = []
        self.image_items = []
        self.canvas.delete("all")
        if not image_paths:
            return
        loaded_imgs = []
        max_height = 0
        for img_path in image_paths:
            try:
                img = Image.open(img_path)
                loaded_imgs.append(img)
                max_height = max(max_height, img.height)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        x_offset = 10
        padding = 10
        for img in loaded_imgs:
            # Align to bottom
            y_offset = max_height - img.height + padding
            self.original_images.append((img, x_offset, y_offset))
            x_offset += img.width + padding
        self.zoom_level = 1.0
        self.update_display()
        
    def update_display(self):
        self.canvas.delete("all")
        self.image_refs = []
        self.image_items = []
        
        for img, x, y in self.original_images:
            # Apply zoom
            new_width = int(img.width * self.zoom_level)
            new_height = int(img.height * self.zoom_level)
            
            if new_width > 0 and new_height > 0:
                zoomed_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(zoomed_img)
                self.image_refs.append(photo)
                
                # Position with zoom
                scaled_x = int(x * self.zoom_level)
                scaled_y = int(y * self.zoom_level)
                
                item = self.canvas.create_image(scaled_x, scaled_y, anchor="nw", image=photo)
                self.image_items.append(item)
        
        # Update scroll region
        self.canvas.update_idletasks()
        bbox = self.canvas.bbox("all")
        if bbox:
            self.canvas.configure(scrollregion=bbox)
            
    def zoom_in(self):
        self.zoom_level = min(self.zoom_level * 1.1, 5.0)
        self.update_display()
        
    def zoom_out(self):
        self.zoom_level = max(self.zoom_level / 1.1, 0.2)
        self.update_display()
        
    def fit_to_window(self):
        if not self.original_images:
            return
        # Calculate total width of all images + padding
        total_width = sum(img.width for img, _, _ in self.original_images)
        total_width += 10 * (len(self.original_images) + 1)
        canvas_width = self.canvas.winfo_width()
        if canvas_width <= 1:  # Not yet rendered
            self.parent.update_idletasks()
            canvas_width = self.canvas.winfo_width()
        if total_width > 0 and canvas_width > 0:
            self.zoom_level = min(1.0, canvas_width / total_width)
        else:
            self.zoom_level = 1.0
        self.update_display()

class UpSetGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("UpSet Plot Generator")
        self.root.geometry("1400x800")
        
        self.csv_file = None
        self.output_images = []
        self.color_mapping = {}
        self.use_colors = tk.BooleanVar(value=True)
        self.group_by_subletters = tk.BooleanVar(value=False)
        self.group_by_words = tk.BooleanVar(value=True)
        self.group_by_same_color = tk.BooleanVar(value=False)
        
        # New variables for refined color grouping
        self.agg_use_colors = tk.BooleanVar(value=True)
        self.agg_group_by_subletters = tk.BooleanVar(value=False)
        self.agg_group_by_words = tk.BooleanVar(value=True)
        self.agg_enable_fuzzy = tk.BooleanVar(value=True)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(5, weight=1)  # Make results frame expand
        
        # --- Merge CSVs section ---
        merge_frame = ttk.LabelFrame(main_frame, text="Merge CSV Files", padding="5")
        merge_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0,2))
        self.merge_label = ttk.Label(merge_frame, text="No files selected", foreground="gray")
        self.merge_label.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5), pady=2)
        # Place the buttons below the label, natural width
        merge_buttons_frame = ttk.Frame(merge_frame)
        merge_buttons_frame.grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        self.merge_button = ttk.Button(merge_buttons_frame, text="Select & Merge CSVs with One Varying Parameter", command=self.merge_csv_files)
        self.merge_button.grid(row=0, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        self.single_csv_button = ttk.Button(merge_buttons_frame, text="Select Single CSV with Multiple Varying Parameters", command=self.process_single_csv_with_params)
        self.single_csv_button.grid(row=0, column=1, sticky=tk.W, padx=(5, 5), pady=2)
        self.full_analysis_button = ttk.Button(merge_buttons_frame, text="Generate Full Analysis", command=self.generate_full_analysis, state='disabled')
        self.full_analysis_button.grid(row=0, column=2, sticky=tk.W, padx=(5, 0), pady=2)
        
        # File selection
        ttk.Label(main_frame, text="CSV File:").grid(row=1, column=0, sticky=tk.W, pady=0)
        self.file_label = ttk.Label(main_frame, text="No file selected", foreground="gray")
        self.file_label.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=0)
        ttk.Button(main_frame, text="Browse", command=self.browse_file).grid(row=1, column=2, padx=(5, 0), pady=0)
        # Block info label (single line, under file label)
        self.block_info_label = ttk.Label(main_frame, text="", foreground="blue")
        self.block_info_label.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=0)
        
        # Options frame
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="3")
        options_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=0)
        
        # Color checkbox and entry
        color_frame = ttk.Frame(options_frame)
        color_frame.grid(row=0, column=0, sticky=tk.W, pady=0)
        self.color_criteria_cb = ttk.Checkbutton(color_frame, text="Color criteria", variable=self.use_colors, command=self.on_color_criteria_toggle)
        self.color_criteria_cb.grid(row=0, column=0, sticky=tk.W, pady=0)
        ttk.Label(color_frame, text="Min shared letters:").grid(row=0, column=1, padx=(5,0), pady=0)
        self.min_letters_var = tk.StringVar(value="5")
        self.min_letters_entry = ttk.Entry(color_frame, textvariable=self.min_letters_var, width=3)
        self.min_letters_entry.grid(row=0, column=2, padx=(2,0), pady=0)
        ttk.Checkbutton(color_frame, text="Group by subletters (except suffixes)", variable=self.group_by_subletters, command=self.on_group_by_subletters).grid(row=0, column=3, padx=(5,0), pady=0)
        ttk.Checkbutton(color_frame, text="Group by whole words", variable=self.group_by_words, command=self.on_group_by_words).grid(row=0, column=4, padx=(5,0), pady=0)
        self.group_by_same_color_cb = ttk.Checkbutton(color_frame, text="Collapse into same color group", variable=self.group_by_same_color)
        self.group_by_same_color_cb.grid(row=0, column=5, padx=(5,0), pady=0)
        self.group_by_same_color_cb.state(['disabled'])
        # Add label wrap width entry
        ttk.Label(color_frame, text="Label wrap width:").grid(row=0, column=6, padx=(5,0), pady=0)
        self.wrap_width_var = tk.StringVar(value="75")
        self.wrap_width_entry = ttk.Entry(color_frame, textvariable=self.wrap_width_var, width=3)
        self.wrap_width_entry.grid(row=0, column=7, padx=(2,0), pady=0)
        
        # Process button and progress
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=0)
        button_frame.columnconfigure(0, weight=1)
        
        self.progress = ttk.Progressbar(button_frame, mode='indeterminate')
        self.progress.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5), pady=0)
        
        self.process_btn = ttk.Button(button_frame, text="Generate UpSet Plots", 
                                     command=self.start_processing, state='disabled')
        self.process_btn.grid(row=0, column=1, pady=0)
        
        # --- Aggregate Results section ---
        aggregate_frame = ttk.LabelFrame(main_frame, text="Aggregate Results from Multiple Folders", padding="5")
        aggregate_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10,2))
        self.aggregate_folder = None
        self.aggregate_folder_label = ttk.Label(aggregate_frame, text="No folder selected", foreground="gray")
        self.aggregate_folder_label.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5), pady=2)
        self.select_aggregate_folder_btn = ttk.Button(aggregate_frame, text="Select Parent Folder", command=self.select_aggregate_folder)
        self.select_aggregate_folder_btn.grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        self.aggregate_btn = ttk.Button(aggregate_frame, text="Aggregate Results", command=self.aggregate_results, state='disabled')
        self.aggregate_btn.grid(row=1, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        
        # Embedding model selection
        ttk.Label(aggregate_frame, text="Embedding Model:").grid(row=1, column=2, sticky=tk.W, padx=(10, 5), pady=2)
        self.embedding_model_var = tk.StringVar(value="🤗 sentence-transformers/all-MiniLM-L6-v2 (384 dim, fast)")
        self.embedding_model_combo = ttk.Combobox(aggregate_frame, textvariable=self.embedding_model_var, 
                                                state="readonly", width=50)
        self.embedding_model_combo['values'] = [
            "🤗 sentence-transformers/all-MiniLM-L6-v2 (384 dim, fast)",
            "🤗 sentence-transformers/all-mpnet-base-v2 (768 dim, high-quality)",
            "🤗 sentence-transformers/all-distilroberta-v1 (768 dim, balanced)",
            "🤗 sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384 dim, multilingual)",
            "🤗 sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (768 dim, multilingual)",
            "🤗 BAAI/bge-small-en-v1.5 (384 dim, efficient)",
            "🤗 BAAI/bge-base-en-v1.5 (768 dim, excellent)",
            "🤗 BAAI/bge-large-en-v1.5 (1024 dim, powerful)",
            "🤗 intfloat/e5-base-v2 (768 dim, general-purpose)",
            "🤗 intfloat/e5-large-v2 (1024 dim, advanced)",
            "🟦 Qwen/Qwen3-Embedding-8B (1024 dim, advanced)",
            "🟦 BAAI/bge-en-icl (1024 dim, instruction-tuned)",
            "🟦 BAAI/bge-multilingual-gemma2 (1024 dim, multilingual)"
        ]
        self.embedding_model_combo.grid(row=1, column=3, sticky=(tk.W, tk.E), padx=(0, 5), pady=2)
        
        # RAG Consistency Analysis Options
        consistency_frame = ttk.Frame(aggregate_frame)
        consistency_frame.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(5,2))
        
        # Add Quotes checkbox
        self.add_quotes_var = tk.BooleanVar(value=False)
        self.add_quotes_cb = ttk.Checkbutton(consistency_frame, text="Add Quotes", 
                                           variable=self.add_quotes_var)
        self.add_quotes_cb.grid(row=0, column=0, sticky=tk.W, pady=2, padx=(0, 10))
        
        # Main consistency checkbox
        self.analyze_consistency_var = tk.BooleanVar(value=False)
        self.analyze_consistency_cb = ttk.Checkbutton(consistency_frame, text="Analyze Consistency", 
                                                    variable=self.analyze_consistency_var, 
                                                    command=self.on_consistency_toggle)
        self.analyze_consistency_cb.grid(row=0, column=1, sticky=tk.W, pady=2)
        
        # Sub-checkboxes for individual analyses
        self.consistency_sub_frame = ttk.Frame(consistency_frame)
        self.consistency_sub_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(2,0))
        
        self.within_param_var = tk.BooleanVar(value=False)
        self.within_param_cb = ttk.Checkbutton(self.consistency_sub_frame, text="A. Within-Parameter Analysis", 
                                             variable=self.within_param_var, state='disabled')
        self.within_param_cb.grid(row=0, column=0, sticky=tk.W, padx=(20, 10), pady=2)
        
        self.cross_param_var = tk.BooleanVar(value=False)
        self.cross_param_cb = ttk.Checkbutton(self.consistency_sub_frame, text="B. Cross-Parameter Analysis", 
                                            variable=self.cross_param_var, state='disabled')
        self.cross_param_cb.grid(row=0, column=1, sticky=tk.W, padx=(10, 10), pady=2)
        
        self.sensitivity_var = tk.BooleanVar(value=False)
        self.sensitivity_cb = ttk.Checkbutton(self.consistency_sub_frame, text="C. Parameter Sensitivity Ranking", 
                                            variable=self.sensitivity_var, state='disabled')
        self.sensitivity_cb.grid(row=0, column=2, sticky=tk.W, padx=(10, 10), pady=2)
        
        self.semantic_viz_var = tk.BooleanVar(value=False)
        self.semantic_viz_cb = ttk.Checkbutton(self.consistency_sub_frame, text="D. Semantic Similarity Across Authors / Folders", 
                                             variable=self.semantic_viz_var, state='disabled',
                                             command=self.on_semantic_viz_toggle)
        self.semantic_viz_cb.grid(row=0, column=3, sticky=tk.W, padx=(10, 0), pady=2)
        
        # t-SNE Configuration Frame (shown when D is checked)
        self.tsne_config_frame = ttk.LabelFrame(consistency_frame, text="t-SNE Configuration (Part D)")
        self.tsne_config_frame.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), padx=20, pady=5)
        
        # Font size
        ttk.Label(self.tsne_config_frame, text="Font Size:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.tsne_font_size_var = tk.IntVar(value=5)
        ttk.Spinbox(self.tsne_config_frame, from_=3, to=20, textvariable=self.tsne_font_size_var, width=5).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Dimensions
        ttk.Label(self.tsne_config_frame, text="Dimensions:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.tsne_n_components_var = tk.IntVar(value=2)
        ttk.Spinbox(self.tsne_config_frame, from_=2, to=3, textvariable=self.tsne_n_components_var, width=5).grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        
        # Color palette
        ttk.Label(self.tsne_config_frame, text="Color Palette:").grid(row=0, column=4, sticky=tk.W, padx=5, pady=2)
        self.tsne_color_palette_var = tk.StringVar(value="Set3")
        color_palettes = ['Set3', 'Set1', 'Set2', 'tab10', 'tab20', 'viridis', 'plasma', 'inferno', 'magma', 'coolwarm']
        ttk.Combobox(self.tsne_config_frame, textvariable=self.tsne_color_palette_var, values=color_palettes, 
                    state="readonly", width=12).grid(row=0, column=5, sticky=tk.W, padx=5, pady=2)
        
        # Proximity threshold
        ttk.Label(self.tsne_config_frame, text="Proximity Grouping:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.tsne_proximity_var = tk.DoubleVar(value=0.0)
        proximity_spinbox = ttk.Spinbox(self.tsne_config_frame, from_=0.0, to=10.0, increment=0.5, 
                                        textvariable=self.tsne_proximity_var, width=5)
        proximity_spinbox.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Label(self.tsne_config_frame, text="(0 = disabled)").grid(row=1, column=2, sticky=tk.W, padx=2, pady=2)
        
        # Show labels
        self.tsne_show_labels_var = tk.BooleanVar(value=True)
        self.tsne_show_labels_cb = ttk.Checkbutton(self.tsne_config_frame, text="Show Concept Labels", 
                       variable=self.tsne_show_labels_var)
        self.tsne_show_labels_cb.grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)
        
        # Store references to widgets that need to be enabled/disabled
        self.tsne_config_widgets = []
        for widget in self.tsne_config_frame.winfo_children():
            if isinstance(widget, (ttk.Spinbox, ttk.Combobox, ttk.Checkbutton)):
                self.tsne_config_widgets.append(widget)
        
        # Initially disable all config widgets
        for widget in self.tsne_config_widgets:
            try:
                widget.config(state='disabled')
            except:
                pass
        
        self.aggregate_status_label = ttk.Label(aggregate_frame, text="", foreground="blue")
        self.aggregate_status_label.grid(row=3, column=0, columnspan=4, sticky=(tk.W, tk.E), padx=(0, 5), pady=2)
        # --- Progress bar for aggregation ---
        self.aggregate_progress = ttk.Progressbar(aggregate_frame, mode='determinate', length=300)
        self.aggregate_progress.grid(row=4, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(2,2))
        self.aggregate_time_label = ttk.Label(aggregate_frame, text="", foreground="gray")
        self.aggregate_time_label.grid(row=5, column=0, columnspan=4, sticky=(tk.W, tk.E), padx=(0, 5), pady=2)
        
        # --- LLM Grouping Controls ---
        llm_grouping_frame = ttk.Frame(aggregate_frame)
        llm_grouping_frame.grid(row=6, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(5,2))
        self.llm_grouping_var = tk.BooleanVar(value=False)
        self.llm_grouping_cb = ttk.Checkbutton(llm_grouping_frame, text="LLM Grouping (semantic)", variable=self.llm_grouping_var, command=self.on_llm_grouping_toggle)
        self.llm_grouping_cb.grid(row=0, column=0, sticky=tk.W, pady=0)
        self.llm_model_var = tk.StringVar(value="Mistral-API")
        self.llm_model_dropdown = ttk.Combobox(llm_grouping_frame, textvariable=self.llm_model_var, values=[
            "Remote Meta-Llama-3", "Mistral-API", "GPT-3.5", "GPT-4o", "GPT-4o mini", "o1-mini", "o3-mini",
            "Gemini", "Claude", "Grok", "Qwen3", "Phi4", "Meta Llama 70B", "DeepSeek V3", "Mistral (Nebius)"
        ], state="readonly", width=18)
        self.llm_model_dropdown.grid(row=0, column=1, padx=(10,0), pady=0)
        self.llm_prompt_var = tk.StringVar(value="Group the following concepts by meaning. Return the result as JSON, where each group is a list of concepts.\nConcepts:\n- concept1\n- concept2\n...")
        self.llm_prompt_entry = ttk.Entry(llm_grouping_frame, textvariable=self.llm_prompt_var, width=60)
        self.llm_prompt_entry.grid(row=0, column=2, padx=(10,0), pady=0)
        # --- Refined Color Grouping Controls ---
        refined_grouping_frame = ttk.Frame(aggregate_frame)
        refined_grouping_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5,2))
        
        # Color grouping options
        agg_color_frame = ttk.Frame(refined_grouping_frame)
        agg_color_frame.grid(row=0, column=0, sticky=tk.W, pady=2)
        self.agg_color_criteria_cb = ttk.Checkbutton(agg_color_frame, text="Refined Color Grouping", variable=self.agg_use_colors, command=self.on_agg_color_criteria_toggle)
        self.agg_color_criteria_cb.grid(row=0, column=0, sticky=tk.W, pady=0)
        ttk.Label(agg_color_frame, text="Min shared letters:").grid(row=0, column=1, padx=(5,0), pady=0)
        self.agg_min_letters_var = tk.StringVar(value="5")
        self.agg_min_letters_entry = ttk.Entry(agg_color_frame, textvariable=self.agg_min_letters_var, width=3)
        self.agg_min_letters_entry.grid(row=0, column=2, padx=(2,0), pady=0)
        ttk.Checkbutton(agg_color_frame, text="Group by subletters (except suffixes)", variable=self.agg_group_by_subletters, command=self.on_agg_group_by_subletters).grid(row=0, column=3, padx=(5,0), pady=0)
        ttk.Checkbutton(agg_color_frame, text="Group by whole words", variable=self.agg_group_by_words, command=self.on_agg_group_by_words).grid(row=0, column=4, padx=(5,0), pady=0)
        
        # Fuzzy logic controls
        fuzzy_frame = ttk.Frame(refined_grouping_frame)
        fuzzy_frame.grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(fuzzy_frame, text="Enable Fuzzy Logic", variable=self.agg_enable_fuzzy, command=self.on_agg_enable_fuzzy_toggle).grid(row=0, column=0, sticky=tk.W, pady=0)
        ttk.Label(fuzzy_frame, text="Similarity Threshold:").grid(row=0, column=1, padx=(5,0), pady=0)
        self.sim_threshold_var = tk.DoubleVar(value=0.85)
        self.sim_threshold_entry = ttk.Entry(fuzzy_frame, textvariable=self.sim_threshold_var, width=5)
        self.sim_threshold_entry.grid(row=0, column=2, sticky=tk.W, padx=(2,10))
        ttk.Label(fuzzy_frame, text="Grouping Logic:").grid(row=0, column=3, sticky=tk.W)
        self.grouping_logic_var = tk.StringVar(value="Fuzzy")
        self.grouping_logic_combo = ttk.Combobox(fuzzy_frame, textvariable=self.grouping_logic_var, values=["Fuzzy", "Exact", "None"], state="readonly", width=8)
        self.grouping_logic_combo.grid(row=0, column=4, sticky=tk.W, padx=(2,0))
        # Add fuzzy logic explanation label
        self.fuzzy_explanation_label = ttk.Label(fuzzy_frame, text="Fuzzy logic groups concepts by overall string similarity.\nThreshold 1.0 = only identical concepts.\nThreshold 0.85 = minor spelling/word order differences.\nThreshold 0.6 = allows more distant matches.\nExample: 'justice' and 'justices' are grouped at 0.85, but 'justice' and 'injustice' only at 0.6.", justify='left', foreground='gray')
        self.fuzzy_explanation_label.grid(row=1, column=0, columnspan=5, sticky=tk.W, pady=(2,0))
        
        # Initialize fuzzy controls state
        self.on_agg_enable_fuzzy_toggle()
        
        # Results frame with zoom controls
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="5")
        results_frame.grid(row=9, column=0, columnspan=3, sticky=(tk.N, tk.S, tk.E, tk.W), pady=5)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        # Zoom controls
        zoom_frame = ttk.Frame(results_frame)
        zoom_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 2))
        
        ttk.Button(zoom_frame, text="Zoom In", command=self.zoom_in).grid(row=0, column=0, padx=(0, 2), pady=0)
        ttk.Button(zoom_frame, text="Zoom Out", command=self.zoom_out).grid(row=0, column=1, padx=(0, 2), pady=0)
        ttk.Button(zoom_frame, text="Fit to Window", command=self.fit_to_window).grid(row=0, column=2, padx=(0, 2), pady=0)
        
        ttk.Label(zoom_frame, text="Use mouse wheel to zoom, click and drag to pan").grid(row=0, column=3, padx=(10, 0), pady=0)
        
        # Zoomable canvas
        canvas_frame = ttk.Frame(results_frame)
        canvas_frame.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        self.zoomable_canvas = ZoomablePanCanvas(canvas_frame)
        self.zoomable_canvas.canvas.grid(sticky=(tk.N, tk.S, tk.E, tk.W))  # Ensure canvas expands
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.grid(row=10, column=0, columnspan=3, pady=2)
        
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            self.csv_file = file_path
            self.file_label.config(text=os.path.basename(file_path), foreground="black")
            self.process_btn.config(state='normal')
            # Try to read and display block info
            self.display_block_info()
            
    def display_block_info(self):
        # Try to read the CSV and infer block structure
        if not self.csv_file:
            self.block_info_label.config(text="", foreground="blue")
            return
        encodings_to_try = ['cp1253', 'utf-8', 'windows-1252']
        df = None
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(self.csv_file, encoding=enc)
                break
            except Exception:
                continue
        if df is None:
            self.block_info_label.config(text="Could not read CSV to infer blocks.", foreground="red")
            return
        
        # Detect parameter columns
        param_names = [col for col in ['Temperature', 'Top-p', 'Top-k', 'BM25 Weight'] if col in df.columns]
        
        if len(param_names) != 4:
            block_info = f"Found {len(param_names)} parameter columns, need 4."
            self.block_info_label.config(text=block_info, foreground="red")
            return
        
        # Analyze parameter values
        param_values = {}
        for param in param_names:
            unique_vals = sorted(df[param].unique())
            param_values[param] = unique_vals
        
        # Check if we have 5 values for each parameter (5x5x5x5 = 625)
        expected_combinations = 1
        for param, values in param_values.items():
            expected_combinations *= len(values)
        
        if expected_combinations != 625:
            block_info = f"Expected 625 combinations (5x5x5x5), but calculated {expected_combinations}"
            self.block_info_label.config(text=block_info, foreground="red")
            return
        
        # For 5x5x5x5 combinations, create proper blocks
        # The data should be organized as: 5 blocks of 125 rows each for the outermost parameter
        blocks = []
        block_strs = []
        
        # Calculate block sizes for each parameter level
        # For a 5x5x5x5 grid:
        # - Outermost parameter: 5 blocks of 125 rows each
        # - Second parameter: 25 blocks of 25 rows each  
        # - Third parameter: 125 blocks of 5 rows each
        # - Innermost parameter: 625 blocks of 1 row each
        
        block_size_125 = 125  # 5^3
        block_size_25 = 25    # 5^2
        block_size_5 = 5      # 5^1
        block_size_1 = 1      # 5^0
        
        # Create blocks for each parameter level
        current_row = 0
        
        # Level 1: 5 blocks of 125 rows each (outermost parameter)
        for i in range(5):
            start = current_row
            end = current_row + block_size_125
            param_name = param_names[0]  # Outermost parameter
            blocks.append((start, end, param_name))
            block_strs.append(f"{start}-{end-1}: {param_name}")
            current_row = end
        
        # Level 2: 25 blocks of 25 rows each (second parameter)
        current_row = 0
        for i in range(25):
            start = current_row
            end = current_row + block_size_25
            param_name = param_names[1]  # Second parameter
            blocks.append((start, end, param_name))
            block_strs.append(f"{start}-{end-1}: {param_name}")
            current_row = end
        
        # Level 3: 125 blocks of 5 rows each (third parameter)
        current_row = 0
        for i in range(125):
            start = current_row
            end = current_row + block_size_5
            param_name = param_names[2]  # Third parameter
            blocks.append((start, end, param_name))
            block_strs.append(f"{start}-{end-1}: {param_name}")
            current_row = end
        
        # Level 4: 625 blocks of 1 row each (innermost parameter)
        current_row = 0
        for i in range(625):
            start = current_row
            end = current_row + block_size_1
            param_name = param_names[3]  # Innermost parameter
            blocks.append((start, end, param_name))
            block_strs.append(f"{start}-{end-1}: {param_name}")
            current_row = end
        
        block_info = "Blocks: [" + ", ".join(block_strs) + "]"
        self.block_info_label.config(text=block_info, foreground="blue")
        self.detected_blocks = blocks
            
    def zoom_in(self):
        self.zoomable_canvas.zoom_in()
        
    def zoom_out(self):
        self.zoomable_canvas.zoom_out()
        
    def fit_to_window(self):
        self.zoomable_canvas.fit_to_window()
        
    def _is_valid_concept(self, concept):
        """Check if a concept is valid (not a number or technical parameter)"""
        if not concept or not isinstance(concept, str):
            return False
        
        concept = concept.strip()
        
        # Filter out empty or very short concepts
        if len(concept) < 2:
            return False
        
        # Filter out pure numbers
        if concept.isdigit():
            return False
        
        # Filter out common parameters and technical terms
        invalid_terms = {
            'bm25', 'temperature', 'top_p', 'top_k', 'top-p', 'top-k', 'block', 'blocks', 'group label', 'index', 'parameter'
        }
        
        # Check for exact matches
        if concept.lower() in invalid_terms:
            return False
        
        # Check if concept contains any invalid terms (like "BM25 Weight", "Top-k", "Top-p")
        concept_lower = concept.lower()
        for invalid_term in invalid_terms:
            if invalid_term in concept_lower:
                return False
        
        # Filter out long descriptive sentences (likely not concepts)
        if len(concept) > 100:
            return False
        
        # Filter out sentences that look like descriptions (contain common sentence words)
        descriptive_words = ['this', 'table', 'shows', 'specific', 'citations', 'quotes', 'original', 'texts', 'concept', 'extracted', 'parameter', 'files', 'demonstrate', 'used', 'context', 'within', 'philosophical']
        concept_words = concept.lower().split()
        if len(concept_words) > 5 and any(word in descriptive_words for word in concept_words):
            return False
        
        return True

    def extract_concepts(self, text):
        if pd.isna(text):
            return []
        lines = text.split('\n')
        concepts = []
        
        # Detect if this is GPT_OSS_120b format (has table structure)
        is_gpt_format = '| # |' in text and '| Concept' in text
        
        for line in lines:
            line = line.strip()
            
            # Handle numbered lists (1., 2., etc.) - for DeepSeekV3 format
            if line.startswith(tuple(str(i)+'.' for i in range(1, 21))):
                # Markdown or plain numbered list
                if '**' in line:
                    c = line.split('**')[1].replace(':', '').replace('.', '').strip()
                    if c and self._is_valid_concept(c):
                        concepts.append(c)
                else:
                    c = line.lstrip('0123456789. ').replace(':', '').replace('.', '').strip()
                    if c and self._is_valid_concept(c):
                        concepts.append(c)
            
            # Handle markdown table format (| # | Concept | ...) - for GPT_OSS_120b format
            # Only extract from numbered table rows (| 1 |, | 2 |, etc.)
            elif line.startswith('|') and '**' in line and any(line.startswith(f'| {i} |') for i in range(1, 21)):
                # Split by | and look for **bold** text in the second column only
                parts = line.split('|')
                if len(parts) >= 3:  # Ensure we have at least 3 parts: | number | concept | description |
                    concept_part = parts[2].strip()  # Second column should contain the concept
                    # Only extract if the entire concept_part is wrapped in ** (not just contains **)
                    if concept_part.startswith('**') and concept_part.endswith('**'):
                        c = concept_part[2:-2].strip()  # Remove ** from both ends
                        if c and c not in ['#', 'Concept', 'How the text uses', 'Where the term appears', 'Why it functions as', 'Why it qualifies as', 'Why it counts as', 'Why it ranks among'] and self._is_valid_concept(c):
                            concepts.append(c)
            
            # No fallback logic needed - only extract from numbered lines and table rows
            # This ensures we only extract concepts from lines that start with a number
        
        return concepts

    def extract_concepts_and_quotes(self, text):
        """Extract both concepts and their associated quotes from text"""
        if pd.isna(text):
            return [], {}
        
        lines = text.split('\n')
        concepts = []
        concept_quotes = {}
        
        # Detect if this is GPT_OSS_120b format (has table structure)
        is_gpt_format = '| # |' in text and '| Concept' in text
        
        for line in lines:
            line = line.strip()
            
            # Handle numbered lists (1., 2., etc.) - for DeepSeekV3 format
            if line.startswith(tuple(str(i)+'.' for i in range(1, 21))):
                # Markdown or plain numbered list
                if '**' in line:
                    c = line.split('**')[1].replace(':', '').replace('.', '').strip()
                    if c and self._is_valid_concept(c):
                        concepts.append(c)
                        # Extract quotes for this concept
                        quotes = self._extract_quotes_for_concept_from_text(text, c)
                        if quotes:
                            concept_quotes[c] = quotes
                            print(f"[QUOTES DEBUG] Found {len(quotes)} quotes for concept '{c}': {quotes}")
                else:
                    c = line.lstrip('0123456789. ').replace(':', '').replace('.', '').strip()
                    if c and self._is_valid_concept(c):
                        concepts.append(c)
                        # Extract quotes for this concept
                        quotes = self._extract_quotes_for_concept_from_text(text, c)
                        if quotes:
                            concept_quotes[c] = quotes
                            print(f"[QUOTES DEBUG] Found {len(quotes)} quotes for concept '{c}': {quotes}")
            
            # Handle markdown table format (| # | Concept | ...) - for GPT_OSS_120b format
            # Only extract from numbered table rows (| 1 |, | 2 |, etc.)
            elif line.startswith('|') and '**' in line and any(line.startswith(f'| {i} |') for i in range(1, 21)):
                # Split by | and look for **bold** text in the second column only
                parts = line.split('|')
                if len(parts) >= 3:  # Ensure we have at least 3 parts: | number | concept | description |
                    concept_part = parts[2].strip()  # Second column should contain the concept
                    # Only extract if the entire concept_part is wrapped in ** (not just contains **)
                    if concept_part.startswith('**') and concept_part.endswith('**'):
                        c = concept_part[2:-2].strip()  # Remove ** from both ends
                        if c and c not in ['#', 'Concept', 'How the text uses', 'Where the term appears', 'Why it functions as', 'Why it qualifies as', 'Why it counts as', 'Why it ranks among'] and self._is_valid_concept(c):
                            concepts.append(c)
                            # Extract quotes for this concept
                            quotes = self._extract_quotes_for_concept_from_text(text, c)
                            if quotes:
                                concept_quotes[c] = quotes
                                print(f"[QUOTES DEBUG] Found {len(quotes)} quotes for concept '{c}': {quotes}")
        
        print(f"[QUOTES DEBUG] Total concepts extracted: {len(concepts)}, quotes found for {len(concept_quotes)} concepts")
        return concepts, concept_quotes

    def _extract_quotes_for_concept_from_text(self, text, concept_name):
        """Extract quotes for a specific concept from the full text"""
        quotes = []
        
        try:
            import re
            
            print(f"[QUOTES DEBUG] Looking for quotes for concept '{concept_name}' in text of length {len(text)}")
            
            # Try multiple patterns to find the concept and its quotes
            patterns_to_try = [
                # Pattern: **Concept** ... *Specific Use*: "quote" (most specific)
                rf'\*\*{re.escape(concept_name)}\*\*.*?\*Specific Use\*:\s*"([^"]+)"',
                # Pattern: **Concept** ... *Specific Use*: quote (without quotes)
                rf'\*\*{re.escape(concept_name)}\*\*.*?\*Specific Use\*:\s*([^"]+?)(?:\n|$)',
                # Pattern: **Concept** ... *Specific Use*: In Book III, Aristotle... (specific format)
                rf'\*\*{re.escape(concept_name)}\*\*.*?\*Specific Use\*:\s*In[^"]+',
                # Pattern: **Concept** ... "quote" (any quoted text after concept)
                rf'\*\*{re.escape(concept_name)}\*\*.*?"([^"]+)"',
                # Pattern: Concept ... "quote" (without **)
                rf'{re.escape(concept_name)}.*?"([^"]+)"',
            ]
            
            for i, pattern in enumerate(patterns_to_try):
                matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
                print(f"[QUOTES DEBUG] Pattern {i+1} found {len(matches)} matches")
                for match in matches:
                    quote = match.strip()
                    if len(quote) > 10:  # Filter out very short quotes
                        quotes.append(quote)
                        print(f"[QUOTES DEBUG] Added quote: {quote[:100]}...")
                
                if quotes:  # If we found quotes, stop trying other patterns
                    break
            
            # If still no quotes, try a more general approach
            if not quotes:
                print(f"[QUOTES DEBUG] No quotes found with specific patterns, trying general approach")
                # Look for any text that contains the concept name and has quotes
                concept_words = concept_name.split()
                if len(concept_words) > 0:
                    # Try with the first word of the concept
                    first_word = concept_words[0]
                    pattern = rf'{re.escape(first_word)}.*?"([^"]+)"'
                    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
                    print(f"[QUOTES DEBUG] General pattern with '{first_word}' found {len(matches)} matches")
                    for match in matches:
                        quote = match.strip()
                        if len(quote) > 20:  # Longer minimum for general quotes
                            quotes.append(quote)
                            print(f"[QUOTES DEBUG] Added general quote: {quote[:100]}...")
            
            # Remove duplicates and limit quotes
            quotes = list(dict.fromkeys(quotes))[:3]  # Keep first 3 unique quotes
            print(f"[QUOTES DEBUG] Final quotes for '{concept_name}': {len(quotes)}")
            
        except Exception as e:
            print(f"[QUOTES ERROR] Error extracting quotes for concept '{concept_name}': {e}")
            import traceback
            traceback.print_exc()
        
        return quotes
    
    def normalize_word(self, word):
        """Normalize word by removing plurals and common variations"""
        word = word.lower().strip()
        
        # Handle common plural forms
        if word.endswith('ies') and len(word) > 4:
            return word[:-3] + 'y'
        elif word.endswith('es') and len(word) > 3:
            return word[:-2]
        elif word.endswith('s') and len(word) > 2:
            return word[:-1]
        
        # Handle common suffixes
        suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion', 'ness', 'ment', 'able', 'ible', 'ful', 'less']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        
        return word
    
    def extract_key_words(self, concept):
        """Extract meaningful words from a concept"""
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must'}
        words = re.findall(r'\b\w+\b', concept.lower())
        meaningful_words = [self.normalize_word(word) for word in words if word not in stop_words and len(word) > 2]
        return meaningful_words
    
    def create_color_mapping(self, all_concepts):
        self.color_mapping = {}
        min_letters = 5
        try:
            min_letters = int(self.min_letters_var.get())
        except Exception:
            pass
        all_concepts_sorted = sorted(all_concepts, key=lambda x: x.lower())
        groups = []
        assigned = set()
        group_by_words = self.group_by_words.get()
        for i, concept in enumerate(all_concepts_sorted):
            concept_lc = concept.lower()
            if concept_lc in assigned:
                continue
            group = [concept]
            words1 = set(re.findall(r'\b\w+\b', concept_lc))
            for j, other in enumerate(all_concepts_sorted):
                other_lc = other.lower()
                if i == j or other_lc in assigned:
                    continue
                words2 = set(re.findall(r'\b\w+\b', other_lc))
                found = False
                if group_by_words:
                    # Only group if they share words that meet the minimum length requirement
                    shared_words = words1 & words2
                    if shared_words and any(len(w) >= min_letters for w in shared_words):
                        found = True
                else:
                    for w1 in words1:
                        for w2 in words2:
                            for k in range(len(w1) - min_letters + 1):
                                sub = w1[k:k+min_letters]
                                if sub and sub in w2:
                                    if (w1.endswith(sub) or w2.endswith(sub)) and sub in common_suffixes:
                                        continue
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                if found:
                    group.append(other)
            groups.append(group)
            assigned.update([c.lower() for c in group])
        colors = [
            '#800000', '#FF8C00', '#228B22', '#8B008B', '#A0522D', '#2E8B57', '#9932CC', '#FFD700',
            '#556B2F', '#C71585', '#8B4513', '#20B2AA', '#B22222', '#FF4500', '#6A5ACD', '#D2691E',
            '#006400', '#708090', '#FF6347', '#483D8B', '#000000', '#808000', '#8B0000', '#FF1493',
        ]
        color_idx = 0
        for group in groups:
            color = colors[color_idx % len(colors)]
            for concept in group:
                self.color_mapping[concept] = color
            color_idx += 1
        return self.color_mapping
    
    def create_colored_text_image(self, text, color, font_size=12):
        """Create an image with colored text on white background"""
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Get text dimensions
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Create image with white background
        img = Image.new('RGB', (text_width + 10, text_height + 6), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw text in color
        draw.text((5, 3), text, fill=color, font=font)
        
        return img
    
    def start_processing(self):
        if not self.csv_file:
            messagebox.showerror("Error", "Please select a CSV file first")
            return
            
        # Start processing in a separate thread
        self.process_btn.config(state='disabled')
        self.progress.start()
        self.status_label.config(text="Processing...")
        
        thread = threading.Thread(target=self.process_data)
        thread.daemon = True
        thread.start()
        
    def process_data(self):
        """Process CSV file - determines whether to use merge or single CSV logic"""
        try:
            print('Starting process_data')
            
            # Check if this is a merged CSV or single CSV
            if hasattr(self, 'merged_file_paths') and self.merged_file_paths:
                # This is a merged CSV - use the merge processing logic
                print('Detected merged CSV files, using merge processing logic')
                self.process_merged_csvs()
                return
            
            # Single CSV processing logic
            print('Processing single CSV file')
            
            # Memory monitoring function
            def log_memory_usage(stage):
                try:
                    import psutil
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    print(f"Memory usage at {stage}: {memory_mb:.1f} MB")
                except ImportError:
                    print(f"Memory monitoring not available - psutil not installed")
            
            log_memory_usage("start")
            encodings_to_try = ['utf-8', 'windows-1252', 'cp1253', 'latin1']
            for enc in encodings_to_try:
                try:
                    df = pd.read_csv(self.csv_file, encoding=enc)
                    print(f'CSV loaded with encoding: {enc}')
                    break
                except Exception as e:
                    print(f'Failed to load CSV with encoding {enc}: {e}')
            else:
                raise Exception('Could not read CSV file with any supported encoding (utf-8, cp1253, windows-1252, latin1)')
            print('First 10 Main Answer values:')
            print(df['Main Answer'].head(10).to_list())
            # Extract concepts and quotes together
            concept_quote_results = df['Main Answer'].apply(self.extract_concepts_and_quotes)
            df['Concepts'] = [result[0] for result in concept_quote_results]
            df['Concept_Quotes'] = [result[1] for result in concept_quote_results]
            print('First 10 Concepts values:')
            print(df['Concepts'].head(10).to_list())
            
            # Check dataset size to prevent memory issues
            total_rows = len(df)
            if total_rows > 1000:
                print(f"Warning: Large dataset detected ({total_rows} rows). This may cause memory issues.")
                response = messagebox.askyesno("Large Dataset Warning", 
                                             f"Dataset has {total_rows} rows which may cause memory issues.\n"
                                             "Consider processing a smaller subset or ensure you have sufficient RAM.\n\n"
                                             "Continue anyway?")
                if not response:
                    return
            
            all_concepts = set()
            for concepts in df['Concepts']:
                all_concepts.update(concepts)
            print(f'All unique concepts: {all_concepts}')
            self.create_color_mapping(list(all_concepts))
            print(f'Color mapping: {self.color_mapping}')
            params = ['Temperature', 'Top-p', 'Top-k', 'BM25 Weight']
            param_labels = {'Temperature': 'Temp', 'Top-p': 'Topp', 'Top-k': 'Topk', 'BM25 Weight': 'BM25'}
            outdir = 'compare_gui_output'
            os.makedirs(outdir, exist_ok=True)
            blocks = getattr(self, 'detected_blocks', None)
            if not blocks or len(blocks) == 0:
                n = len(df)
                
                # Use the working approach from the older version
                if n == 625:
                    block_size = 125  # 5^3
                    blocks = [
                        (0, 125, 'Temperature'),      # Rows 0-124: Temperature varies
                        (125, 250, 'Top-p'),          # Rows 125-249: Top-p varies  
                        (250, 375, 'Top-k'),          # Rows 250-374: Top-k varies
                        (375, 500, 'BM25 Weight')     # Rows 375-499: BM25 Weight varies
                    ]
                else:
                    # Fallback for other dataset sizes
                    block_size = n // 4
                    blocks = []
                    for i in range(4):
                        start = i * block_size
                        if i == 3:  # Last block - ensure it covers all remaining data
                            end = n
                        else:
                            end = (i + 1) * block_size
                        blocks.append((start, end, params[i] if i < len(params) else '?'))
                
                print(f'Created {len(blocks)} blocks: {blocks}')
            output_files = []
            for start, end, varying_param in blocks:
                print(f'Processing block: {varying_param} ({start}-{end})')
                log_memory_usage(f"before block {varying_param}")
                subset = df.iloc[start:end].copy()
                if subset.empty:
                    print('Subset empty, skipping')
                    continue
                mlb = MultiLabelBinarizer()
                concept_matrix = pd.DataFrame(mlb.fit_transform(subset['Concepts']), columns=mlb.classes_)
                print(f'Concept matrix columns: {concept_matrix.columns}')
                for p in params:
                    concept_matrix[p] = subset[p].values
                concept_matrix_reset = concept_matrix.drop(params, axis=1).astype(bool).reset_index(drop=True)
                print(f'Concept matrix reset columns: {concept_matrix_reset.columns}')
                if self.use_colors.get() and self.group_by_same_color.get():
                    color_map = {col: self.color_mapping.get(col, None) for col in concept_matrix_reset.columns}
                    print(f'Color map: {color_map}')
                    color_groups = defaultdict(list)
                    for col, color in color_map.items():
                        color_groups[color].append(col)
                    print(f'Color groups: {color_groups}')
                    merged = pd.DataFrame(index=concept_matrix_reset.index)
                    for color, cols in color_groups.items():
                        print(f'Grouping for color {color}: {cols}')
                        if color is None or len(cols) == 0:
                            continue
                        if len(cols) == 1:
                            merged[cols[0]] = concept_matrix_reset[cols[0]]
                        else:
                            group_name = "/".join(cols)
                            print(f'Group name: {group_name}')
                            group_name_wrapped = wrap_label(group_name, width=self.get_wrap_width())
                            merged[group_name_wrapped] = concept_matrix_reset[cols].any(axis=1)
                    concept_matrix_reset = merged
                print(f'Final columns for upset: {concept_matrix_reset.columns}')
                upset_data = from_indicators(concept_matrix_reset, concept_matrix_reset.columns)
                print('UpSet data created')
                fig = plt.figure(figsize=(12, 8))
                upset = UpSet(upset_data, show_counts=True)
                axes = upset.plot(fig=fig)
                bar_ax = axes['intersections']
                matrix_ax = axes['matrix']
                bars = bar_ax.patches
                upset_index = upset_data.index
                wrap_width = self.get_wrap_width()
                # Restore color and label logic
                if self.use_colors.get():
                    yticks = matrix_ax.get_yticklabels()
                    wrapped_labels = []
                    for label in yticks:
                        concept = label.get_text()
                        wrapped = wrap_label(concept, width=wrap_width)
                        wrapped_labels.append(wrapped)
                    matrix_ax.set_yticklabels(wrapped_labels)
                    for label, concept in zip(matrix_ax.get_yticklabels(), [l.get_text().replace('\n', ' ') for l in yticks]):
                        if concept in self.color_mapping:
                            color = self.color_mapping[concept]
                        else:
                            first_concept = concept.split('/')[0]
                            color = self.color_mapping.get(first_concept, 'black')
                        label.set_color(color)
                        label.set_weight('bold')
                        label.set_fontsize(10)
                else:
                    yticks = matrix_ax.get_yticklabels()
                    wrapped_labels = [wrap_label(label.get_text(), width=wrap_width) for label in yticks]
                    matrix_ax.set_yticklabels(wrapped_labels)
                    for label in matrix_ax.get_yticklabels():
                        label.set_color('black')
                        label.set_weight('normal')
                        label.set_fontsize(10)
                # Add parameter value labels in red - Ensure ALL bars get labeled (no filtering by height)
                significant_bars = []
                for i, (bar, intersection) in enumerate(zip(bars, upset_index)):
                    if i == 0:
                        continue
                    if bar.get_height() > 0:
                        significant_bars.append((i, bar, intersection))
                
                for i, bar, intersection in significant_bars:
                    x = bar.get_x() + bar.get_width() / 2 + 1  # Skip first column by adding offset
                    intersection_idx = i - 1
                    actual_intersection = upset_index[intersection_idx]
                    mask = np.ones(len(concept_matrix), dtype=bool)
                    for col, present in zip(concept_matrix_reset.columns, actual_intersection):
                        original_cols = col.split('/') if '/' in col else [col]
                        if present:
                            for orig_col in original_cols:
                                if orig_col in concept_matrix.columns:
                                    mask &= concept_matrix[orig_col] == 1
                        else:
                            for orig_col in original_cols:
                                if orig_col in concept_matrix.columns:
                                    mask &= concept_matrix[orig_col] == 0
                    param_vals = concept_matrix.loc[mask, varying_param].unique()
                    def fmt(v):
                        try:
                            f = float(v)
                            return f"{f:.1f}"
                        except Exception:
                            return str(v)
                    label = ','.join(fmt(v) for v in param_vals) if len(param_vals) > 0 else ''
                    if label:  # Only draw if there's a label
                        print(f"Drawing parameter label: '{label}' at x={x}")
                        # Position the label aligned with the top of the first row
                        y_label = len(concept_matrix_reset.columns) - 0.3  # Position at top of first row
                        matrix_ax.text(x, y_label, label, ha='center', va='bottom', fontsize=8, color='red', rotation=90, clip_on=False, weight='bold')
                # Add title and subtitle
                other_params = [p for p in params if p != varying_param]
                fixed_vals = {param_labels[p]: subset[p].iloc[0] for p in other_params}
                fixed_str = ' | '.join(f"{p}={v}" for p, v in fixed_vals.items())
                plt.suptitle(f"UpSet Diagram: {param_labels[varying_param]} sweep\nOther Params: {fixed_str}", fontsize=14, y=0.98)
                plt.subplots_adjust(top=0.88, bottom=0.12)
                plt.tight_layout(rect=[0, 0.12, 1, 0.88])
                outpath = f"{outdir}/compare_{param_labels[varying_param]}_composed.png"
                try:
                    plt.savefig(outpath, dpi=150, bbox_inches='tight', pad_inches=0.5, format='png')
                except Exception as e:
                    print(f"Error saving PNG diagram: {e}")
                plt.close('all')
                
                # Clean up memory after each block
                import gc
                del concept_matrix, concept_matrix_reset, upset_data, upset, fig, axes, bars, upset_index
                gc.collect()
                log_memory_usage(f"after block {varying_param}")
                
                output_files.append(outpath)
            self.root.after(0, self.update_results, output_files)
            self.generate_html_table(blocks, df, self.color_mapping, outdir, param_labels)
            self.generate_docx_and_csv(blocks, df, self.color_mapping, outdir, param_labels)
            self.generate_stats_files(blocks, df, self.color_mapping, outdir, param_labels)
            self.root.after(0, self.display_block_info)
        except Exception as e:
            print('Exception in process_data:')
            traceback.print_exc()
            self.root.after(0, self.show_error, str(e))
    
    def update_results(self, output_files):
        # Get current window width
        window_width = self.root.winfo_width()
        
        # Update zoomable canvas with new images
        self.zoomable_canvas.set_images(output_files, window_width)
        
        # Stop progress and update status
        self.progress.stop()
        self.process_btn.config(state='normal')
        self.status_label.config(text=f"Processing complete. Generated {len(output_files)} plots.")
        
        # Call fit_to_window after processing
        self.zoomable_canvas.fit_to_window()
    
    def show_error(self, error_msg):
        self.progress.stop()
        self.process_btn.config(state='normal')
        self.status_label.config(text=f"Error occurred: {error_msg}")
        messagebox.showerror("Processing Error", f"An error occurred during processing:\n\n{error_msg}")

    def generate_html_table(self, blocks, df, color_mapping, outdir, param_labels):
        html_lines = [
            "<style>",
            ".concept-box { display: inline-block; padding: 1px 3px; margin: 1px; border-radius: 2px; font-weight: bold; white-space: nowrap; font-size: 10px; color: #fff; }",
            ".parameter-box { font-weight: bold; color: #333; font-size: 11px; }",
            ".varying-parameter { background-color: #ffffcc; }",
            ".concepts-line { white-space: nowrap; overflow-x: auto; }",
            "table { width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 12px; }",
            "th, td { border: 1px solid #ddd; padding: 6px; text-align: left; vertical-align: top; }",
            "th { background-color: #f2f2f2; font-size: 11px; }",
            "</style>",
            "<table>",
            "<thead><tr>"
            "<th class='varying-parameter'>Temperature</th>"
            "<th class='varying-parameter'>Top P</th>"
            "<th class='varying-parameter'>Top K</th>"
            "<th class='varying-parameter'>BM25 Weight</th>"
            "<th>Extracted Concepts</th>"
            "</tr></thead><tbody>"
        ]
        for start, end, varying_param in blocks:
            subset = df.iloc[start:end].copy()
            for idx, row in subset.iterrows():
                html_lines.append("<tr>")
                html_lines.append(f"<td class='varying-parameter'><span class='parameter-box'>{row.get('Temperature','')}</span></td>")
                html_lines.append(f"<td class='varying-parameter'><span class='parameter-box'>{row.get('Top-p','')}</span></td>")
                html_lines.append(f"<td class='varying-parameter'><span class='parameter-box'>{row.get('Top-k','')}</span></td>")
                html_lines.append(f"<td class='varying-parameter'><span class='parameter-box'>{row.get('BM25 Weight','')}</span></td>")
                html_lines.append("<td><div class='concepts-line'>")
                for concept in row['Concepts']:
                    color = color_mapping.get(concept, '#888')
                    html_lines.append(
                        f"<span class='concept-box' style='background-color: {color};'>{html.escape(concept)}</span>"
                    )
                html_lines.append("</div></td></tr>")
        html_lines.append("</tbody></table>")
        with open(os.path.join(outdir, "compare.htm"), "w", encoding="utf-8") as f:
            f.write("\n".join(html_lines))

    def get_wrap_width(self):
        try:
            return int(self.wrap_width_var.get())
        except Exception:
            return 75

    def merge_csv_files(self):
        import re
        import traceback
        file_paths = filedialog.askopenfilenames(
            title="Select CSV files to merge",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not file_paths:
            return
        # Store the folder name for DOCX title
        folder_name = os.path.basename(os.path.dirname(file_paths[0])) if file_paths else ""
        self.merged_folder_name = folder_name
        self.merge_label.config(text="Selected files:\n" + "\n".join([os.path.basename(fp) for fp in file_paths]), foreground="black")
        error_patterns = [
            r"error generating response:",
            r"api error occurred:",
            r"bad gateway",
            r"cloudflare",
            r"server disconnected without sending a response",
            r"getaddrinfo failed"
        ]
        error_found = False
        error_msgs = []
        dfs = []
        non_utf8_files = []
        for fp in file_paths:
            df = None
            last_exc = None
            used_encoding = None
            try:
                df = pd.read_csv(fp, encoding='utf-8')
                used_encoding = 'utf-8'
            except Exception as e:
                last_exc = e
                non_utf8_files.append(os.path.basename(fp))
                print(f"File {os.path.basename(fp)} could not be read as utf-8: {e}")
            if used_encoding and used_encoding != 'utf-8':
                print(f"Warning: File {os.path.basename(fp)} was read with encoding {used_encoding}, not utf-8. This may cause issues with special characters.")
            if df is None:
                error_found = True
                error_msgs.append(f"File: {os.path.basename(fp)}, Error reading file as utf-8: {last_exc}")
                continue
            for idx, row in df.iterrows():
                for col in df.columns:
                    val = str(row[col]).lower()
                    for pat in error_patterns:
                        if re.search(pat, val):
                            error_found = True
                            error_msgs.append(f"File: {os.path.basename(fp)}, Row: {idx+2}, Column: '{col}', Error: {row[col]}")
            dfs.append(df)
        if non_utf8_files:
            msg = "Merge aborted: The following files are not valid UTF-8 and may cause encoding issues. Please convert them to UTF-8 and try again:\n" + "\n".join(non_utf8_files)
            self.merge_label.config(text=msg, foreground="red")
            messagebox.showerror("Merge Error", msg)
            return
        if error_found:
            msg = "Merge aborted due to invalid data in the following locations:\n" + "\n".join(error_msgs)
            self.merge_label.config(text=msg, foreground="red")
            messagebox.showerror("Merge Error", msg)
            return
        try:
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_path = os.path.join(os.getcwd(), "compare_input.csv")
            merged_df.to_csv(merged_path, index=False, encoding='utf-8')
            self.csv_file = merged_path
            self.file_label.config(text=os.path.basename(merged_path), foreground="black")
            self.merge_label.config(text=f"Merged:\n" + "\n".join([os.path.basename(fp) for fp in file_paths]) + f"\n(encoding: utf-8)", foreground="black")
            self.process_btn.config(state='normal')
            
            # Store the file paths for later processing
            self.merged_file_paths = file_paths
        except Exception as e:
            import sys
            import traceback
            tb_str = traceback.format_exc()
            print(f"Failed to merge files: {e}\nTraceback:\n{tb_str}")
            error_detail = f"Failed to merge files: {e}\nFiles attempted: {', '.join([os.path.basename(fp) for fp in file_paths])}\nTraceback (see shell):\n{tb_str}"
            self.merge_label.config(text=error_detail, foreground="red")
            messagebox.showerror("Merge Error", error_detail)
            
            # Store the file paths for later processing
            self.merged_file_paths = file_paths

    def process_merged_csvs(self):
        """Process merged CSV files (for Select & Merge CSVs button)"""
        try:
            print('Starting process_merged_csvs')
            
            if not hasattr(self, 'merged_file_paths') or not self.merged_file_paths:
                print('No merged file paths found')
                return
            
            # Process each CSV file separately to create upset plots
            output_files = []
            outdir = 'compare_gui_output'
            os.makedirs(outdir, exist_ok=True)
            
            # Store processed dataframes to preserve Concept_Quotes column
            processed_dataframes = []
            
            for file_path in self.merged_file_paths:
                # Assign file_name at the very beginning to avoid UnboundLocalError
                file_name = os.path.basename(file_path)
                print(f'Processing file: {file_name}')
                
                # Read the CSV file
                print(f"[DEBUG] Reading CSV file: {file_path}")
                print(f"[DEBUG] File size: {os.path.getsize(file_path)} bytes")
                
                # Try to read with different parameters to debug the issue
                try:
                    # First try with python engine which handles malformed CSV better
                    import csv
                    df = pd.read_csv(file_path, encoding='utf-8', engine='python', quoting=csv.QUOTE_ALL, on_bad_lines='skip')
                    print(f'Loaded {len(df)} rows from {file_name}')
                    print(f"[DEBUG] DataFrame shape: {df.shape}")
                    print(f"[DEBUG] DataFrame columns: {df.columns.tolist()}")
                    
                    # Check if there are any NaN values that might be causing issues
                    print(f"[DEBUG] NaN count in each column: {df.isna().sum().to_dict()}")
                    
                except Exception as e:
                    print(f"[ERROR] Failed to read CSV with python engine: {e}")
                    # Try alternative reading methods
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        print(f"[DEBUG] Raw file has {len(lines)} lines")
                        # Try with different parameters
                        df = pd.read_csv(file_path, encoding='utf-8', engine='python', quoting=csv.QUOTE_ALL, on_bad_lines='skip')
                        print(f'Loaded {len(df)} rows with python engine from {file_name}')
                    except Exception as e2:
                        print(f"[ERROR] Alternative reading also failed: {e2}")
                        raise
                
                # Extract concepts and quotes together
                concept_quote_results = df['Main Answer'].apply(self.extract_concepts_and_quotes)
                df['Concepts'] = [result[0] for result in concept_quote_results]
                df['Concept_Quotes'] = [result[1] for result in concept_quote_results]
                print(f"[DEBUG] Extracted concepts: {df['Concepts'].tolist()}")
                
                # Create color mapping for this file
                all_concepts = set()
                for concepts in df['Concepts']:
                    all_concepts.update(concepts)
                self.create_color_mapping(list(all_concepts))
                
                # Create concept matrix
                from sklearn.preprocessing import MultiLabelBinarizer
                mlb = MultiLabelBinarizer()
                concept_matrix = pd.DataFrame(mlb.fit_transform(df['Concepts']), columns=mlb.classes_)
                
                # Ensure all values are boolean - this is the key fix for upsetplot
                for col in concept_matrix.columns:
                    concept_matrix[col] = concept_matrix[col].astype(bool)
                
                print(f"[DEBUG] Concept matrix shape: {concept_matrix.shape}")
                print(f"[DEBUG] Concept matrix columns: {concept_matrix.columns.tolist()}")
                
                # Determine which parameter is varying in this file
                file_name_lower = file_name.lower()
                varying_param = None
                if 'temp' in file_name_lower:
                    varying_param = 'Temperature'
                elif 'top_p' in file_name_lower or 'topp' in file_name_lower:
                    varying_param = 'Top-p'
                elif 'top_k' in file_name_lower or 'topk' in file_name_lower:
                    varying_param = 'Top-k'
                elif 'bm25' in file_name_lower:
                    varying_param = 'BM25 Weight'
                
                # Define parameter labels for consistent naming
                param_labels = {'Temperature': 'Temp', 'Top-p': 'Topp', 'Top-k': 'Topk', 'BM25 Weight': 'BM25'}
                
                print(f"[DEBUG] Detected varying parameter: {varying_param}")
                if varying_param and varying_param in df.columns:
                    print(f"[DEBUG] Column '{varying_param}' exists in DataFrame")
                    print(f"[DEBUG] Column '{varying_param}' values: {df[varying_param].tolist()}")
                    print(f"[DEBUG] Column '{varying_param}' value counts: {df[varying_param].value_counts().to_dict()}")
                    unique_param_values = sorted(df[varying_param].unique())
                    print(f"[DEBUG] Found {len(unique_param_values)} unique values for {varying_param}: {unique_param_values}")
                else:
                    print(f"[DEBUG] Column '{varying_param}' NOT found in DataFrame")
                    print(f"[DEBUG] Available columns: {df.columns.tolist()}")
                    continue
                
                # Apply color grouping if enabled
                if self.use_colors.get() and self.group_by_same_color.get():
                    from collections import defaultdict
                    color_map = {col: self.color_mapping.get(col, None) for col in concept_matrix.columns}
                    color_groups = defaultdict(list)
                    for col, color in color_map.items():
                        color_groups[color].append(col)
                    
                    merged = pd.DataFrame(index=concept_matrix.index)
                    for color, cols in color_groups.items():
                        if color is None or len(cols) == 0:
                            continue
                        if len(cols) == 1:
                            merged[cols[0]] = concept_matrix[cols[0]]
                        else:
                            group_name = "/".join(cols)
                            group_name_wrapped = wrap_label(group_name, width=self.get_wrap_width())
                            merged[group_name_wrapped] = concept_matrix[cols].any(axis=1)
                    concept_matrix = merged
                
                # Create UpSet plot
                try:
                    from upsetplot import from_indicators, UpSet
                    import matplotlib.pyplot as plt
                    
                    # FIX: Ensure each parameter value gets a unique concept pattern
                    # The issue is that some parameter values have identical concept patterns
                    # We need to make them unique while preserving the original concepts
                    
                    print(f"[DEBUG] Target: Need exactly {len(unique_param_values)} bars")
                    print(f"[DEBUG] Original concept matrix shape: {concept_matrix.shape}")
                    
                    # First check if we have enough unique patterns
                    upset_data = from_indicators(concept_matrix, concept_matrix.columns)
                    print(f"[DEBUG] Initial upset data shape: {upset_data.shape}")
                    
                    # Count unique patterns
                    unique_patterns = set()
                    for pattern in upset_data.index:
                        pattern_tuple = tuple(pattern)
                        unique_patterns.add(pattern_tuple)
                    
                    print(f"[DEBUG] Found {len(unique_patterns)} unique patterns out of {len(upset_data)} total patterns")
                    
                    # ALWAYS modify the matrix to ensure each parameter value gets exactly one bar
                    # This guarantees we have exactly len(unique_param_values) + 1 bars (including empty first column)
                    print(f"[DEBUG] Modifying concept matrix to ensure uniqueness for all parameters")
                    
                    # Create a modified matrix that ensures each parameter value is unique
                    # by duplicating existing concepts instead of adding artificial ones
                    modified_matrix = concept_matrix.copy()
                    
                    # First, identify which parameter values have duplicate patterns
                    pattern_map = {}
                    for i, param_value in enumerate(unique_param_values):
                        param_rows = df[df[varying_param] == param_value]
                        if len(param_rows) > 0:
                            row_idx = param_rows.index[0]
                            # Get the pattern for this parameter value
                            pattern = tuple(concept_matrix.loc[row_idx].values)
                            if pattern in pattern_map:
                                pattern_map[pattern].append((param_value, row_idx))
                            else:
                                pattern_map[pattern] = [(param_value, row_idx)]
                    
                    # Now modify the matrix to ensure uniqueness
                    for pattern, param_list in pattern_map.items():
                        if len(param_list) > 1:
                            # Multiple parameter values share this pattern - make them unique
                            print(f"[DEBUG] Found {len(param_list)} parameter values with identical pattern, making unique")
                            
                            for i, (param_value, row_idx) in enumerate(param_list):
                                # Get the original concepts for this parameter value
                                original_concepts = df.loc[row_idx, 'Concepts']
                                
                                # Choose a concept to duplicate (use the last one to minimize impact)
                                if original_concepts:
                                    concept_to_duplicate = original_concepts[-1]  # Use last concept
                                    duplicated_concept_name = f"{concept_to_duplicate}_{i+1}"
                                    
                                    # Add the duplicated concept to the matrix
                                    if duplicated_concept_name not in modified_matrix.columns:
                                        modified_matrix[duplicated_concept_name] = False
                                    
                                    # Set this duplicated concept to True only for this parameter value
                                    modified_matrix.loc[row_idx, duplicated_concept_name] = True
                                    
                                    # IMPORTANT: Remove the original concept from this parameter value
                                    # to avoid duplicate dots in the same column
                                    if concept_to_duplicate in modified_matrix.columns:
                                        modified_matrix.loc[row_idx, concept_to_duplicate] = False
                                    
                                    print(f"[DEBUG] Made parameter {param_value} unique by replacing '{concept_to_duplicate}' with '{duplicated_concept_name}'")
                        else:
                            # Only one parameter value has this pattern - no modification needed
                            param_value, row_idx = param_list[0]
                            print(f"[DEBUG] Parameter {param_value} already has unique pattern")
                    
                    # Use the modified matrix
                    upset_data = from_indicators(modified_matrix, modified_matrix.columns)
                    print(f"[DEBUG] Modified upset data shape: {upset_data.shape}")
                    
                    # Verify uniqueness
                    unique_patterns_after = set()
                    for pattern in upset_data.index:
                        pattern_tuple = tuple(pattern)
                        unique_patterns_after.add(pattern_tuple)
                    
                    print(f"[DEBUG] After modification: {len(unique_patterns_after)} unique patterns")
                    
                    print(f"[DEBUG] Final upset data shape: {upset_data.shape}")
                    
                    fig = plt.figure(figsize=(12, 8))
                    # Show counts on the histogram
                    upset = UpSet(upset_data, show_counts=True)
                    axes = upset.plot(fig=fig)
                    bar_ax = axes['intersections']
                    matrix_ax = axes['matrix']
                    bars = bar_ax.patches
                    upset_index = upset_data.index
                    
                    print(f"[DEBUG] Number of bars: {len(bars)}")
                    print(f"[DEBUG] Number of upset_index entries: {len(upset_index)}")
                    
                    # Apply color and label logic
                    wrap_width = self.get_wrap_width()
                    if self.use_colors.get():
                        yticks = matrix_ax.get_yticklabels()
                        wrapped_labels = []
                        for label in yticks:
                            concept = label.get_text()
                            wrapped = wrap_label(concept, width=wrap_width)
                            wrapped_labels.append(wrapped)
                        matrix_ax.set_yticklabels(wrapped_labels)
                        for label, concept in zip(matrix_ax.get_yticklabels(), [l.get_text().replace('\n', ' ') for l in yticks]):
                            if concept in self.color_mapping:
                                color = self.color_mapping[concept]
                            else:
                                first_concept = concept.split('/')[0]
                                color = self.color_mapping.get(first_concept, 'black')
                            label.set_color(color)
                            label.set_weight('bold')
                            label.set_fontsize(10)
                    else:
                        yticks = matrix_ax.get_yticklabels()
                        wrapped_labels = [wrap_label(label.get_text(), width=wrap_width) for label in yticks]
                        matrix_ax.set_yticklabels(wrapped_labels)
                        for label in matrix_ax.get_yticklabels():
                            label.set_color('black')
                            label.set_weight('normal')
                            label.set_fontsize(10)
                    
                    # Add parameter value labels in red below the matrix
                    if varying_param and varying_param in df.columns:
                        print(f"[DEBUG] Adding labels for {len(unique_param_values)} parameter values")
                        
                        # Get the actual bar positions from the plot
                        bar_positions = [bar.get_x() + bar.get_width()/2 for bar in bars]
                        print(f"[DEBUG] Bar center positions: {bar_positions}")
                        
                        # Helper function for formatting parameter values
                        def fmt(v):
                            try:
                                f = float(v)
                                return f"{f:.2f}"
                            except Exception:
                                return str(v)
                        
                        # Map parameter values to bar positions
                        # We need to handle cases where upsetplot doesn't create enough bars
                        if len(bar_positions) >= len(unique_param_values):
                            # We have enough bars, map each parameter to a bar
                            sorted_params = sorted(unique_param_values)
                            
                            for i, param_value in enumerate(sorted_params):
                                if i < len(bar_positions):
                                    bar_x = bar_positions[i] + 1  # Skip first column by adding offset
                                    label = fmt(param_value)
                                    print(f"[DEBUG] Drawing parameter label: '{label}' at bar position {bar_x:.2f}")
                                    
                                    # Position the label above the matrix columns (above the first row)
                                    # Get the actual matrix bounds from the plot to position labels correctly
                                    matrix_bounds = matrix_ax.get_ylim()
                                    y_label = matrix_bounds[1] - 0.5  # Position about 0.5 rows below the top of the matrix (above first row)
                                    matrix_ax.text(bar_x, y_label, label, ha='center', va='bottom', 
                                                 fontsize=8, color='red', rotation=90, clip_on=False, weight='bold')
                                else:
                                    print(f"[DEBUG] Warning: No bar available for parameter {param_value}")
                        else:
                            # Not enough bars - ensure ALL parameter values get labeled at fixed positions
                            print(f"[DEBUG] Warning: Not enough bars ({len(bar_positions)}) for parameters ({len(unique_param_values)})")
                            print(f"[DEBUG] Creating labels for ALL parameter values at fixed positions")
                            
                            # Sort parameters
                            sorted_params = sorted(unique_param_values)
                            
                            # Create labels for ALL parameter values at evenly spaced positions
                            for i, param_value in enumerate(sorted_params):
                                # Create evenly spaced positions across the plot width
                                if len(sorted_params) == 1:
                                    bar_x = 2  # Center position
                                else:
                                    # Distribute evenly across a reasonable width
                                    bar_x = 1 + (i * 3)  # Start at 1, space by 3 units
                                
                                label = fmt(param_value)
                                print(f"[DEBUG] Drawing parameter label: '{label}' at fixed position {bar_x:.2f}")
                                
                                # Position the label above the matrix columns (above the first row)
                                # Get the actual matrix bounds from the plot to position labels correctly
                                matrix_bounds = matrix_ax.get_ylim()
                                y_label = matrix_bounds[1] - 0.5  # Position about 0.5 rows below the top of the matrix (above first row)
                                matrix_ax.text(bar_x, y_label, label, ha='center', va='bottom', 
                                             fontsize=8, color='red', rotation=90, clip_on=False, weight='bold')
                    
                    # Add title and subtitle
                    # Always define file_name_clean for the plot filename
                    file_name_clean = file_name.replace('.csv', '')
                    
                    if varying_param:
                        # First line: Parameter name + "Sweep"
                        if varying_param == 'Temperature':
                            first_line = "Temp Sweep"
                        elif varying_param == 'Top-p':
                            first_line = "Top-p Sweep"
                        elif varying_param == 'Top-k':
                            first_line = "Top-k Sweep"
                        elif varying_param == 'BM25 Weight':
                            first_line = "BM25 Sweep"
                        else:
                            first_line = f"{varying_param} Sweep"
                        
                        # Second line: Other parameters that were kept constant
                        other_params = []
                        for param in ['Temperature', 'Top-p', 'Top-k', 'BM25 Weight']:
                            if param != varying_param and param in df.columns:
                                # Get the constant value for this parameter
                                constant_value = df[param].iloc[0]  # All rows should have same value
                                other_params.append(f"{param}={constant_value}")
                        
                        second_line = "Other params: " + ", ".join(other_params)
                        
                        plt.title(f"{first_line}\n{second_line}", fontsize=14)
                    else:
                        plt.title(f"UpSet Plot: {file_name_clean}", fontsize=14)
                    plt.tight_layout()
                    
                    # Save the plot
                    plot_path = os.path.join(outdir, f"compare_{param_labels.get(varying_param, varying_param)}_composed.png")
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight', pad_inches=0.5)
                    plt.close()
                    
                    output_files.append(plot_path)
                    print(f"UpSet plot saved: {plot_path}")
                    
                except Exception as plot_error:
                    print(f"Error creating UpSet plot for {file_name}: {plot_error}")
                    import traceback
                    traceback.print_exc()
                
                # Add file information to the dataframe
                df['File'] = file_name
                print(f"[QUOTES DEBUG] Added File column to dataframe: {file_name}")
                
                # Store the processed dataframe for later use
                processed_dataframes.append(df)
                
                # Clean up memory
                import gc
                gc.collect()
            
            # Update results
            self.root.after(0, self.update_results, output_files)
            
            # Generate additional files (HTML, DOCX, CSV, stats)
            if hasattr(self, 'merged_file_paths') and self.merged_file_paths:
                # Use the processed dataframes that already have Concept_Quotes column
                combined_df = pd.concat(processed_dataframes, ignore_index=True)
                print(f"[QUOTES DEBUG] Combined dataframe has {len(combined_df)} rows and columns: {list(combined_df.columns)}")
                print(f"[QUOTES DEBUG] Concept_Quotes column exists: {'Concept_Quotes' in combined_df.columns}")
                
                # Create file-to-varying-parameter mapping and blocks
                file_to_varying_param = {}
                blocks = []
                current_start = 0
                
                for i, file_path in enumerate(self.merged_file_paths):
                    file_name = os.path.basename(file_path)
                    file_name_lower = file_name.lower()
                    print(f"[QUOTES DEBUG] Analyzing file: {file_name}")
                    
                    # Determine varying parameter for this file
                    if 'temp' in file_name_lower:
                        varying_param = 'Temperature'
                        print(f"[QUOTES DEBUG] Detected Temperature sweep in {file_name}")
                    elif 'top_p' in file_name_lower or 'topp' in file_name_lower:
                        varying_param = 'Top-p'
                        print(f"[QUOTES DEBUG] Detected Top-p sweep in {file_name}")
                    elif 'top_k' in file_name_lower or 'topk' in file_name_lower:
                        varying_param = 'Top-k'
                        print(f"[QUOTES DEBUG] Detected Top-k sweep in {file_name}")
                    elif 'bm25' in file_name_lower:
                        varying_param = 'BM25 Weight'
                        print(f"[QUOTES DEBUG] Detected BM25 sweep in {file_name}")
                    else:
                        varying_param = 'Unknown'
                        print(f"[QUOTES DEBUG] Unknown parameter type in {file_name}")
                    
                    # Map file to varying parameter
                    file_to_varying_param[file_name] = varying_param
                    
                    # Count rows for this file in the combined dataframe
                    file_rows = combined_df[combined_df['File'] == file_name]
                    file_end = current_start + len(file_rows)
                    
                    # Create block for this file
                    blocks.append((current_start, file_end, varying_param))
                    print(f"[QUOTES DEBUG] Created block for {file_name}: rows {current_start}-{file_end-1}, varying param: {varying_param}")
                    
                    current_start = file_end
                
                print(f"[QUOTES DEBUG] File to varying param mapping: {file_to_varying_param}")
                print(f"[QUOTES DEBUG] Created blocks: {blocks}")
                
                # Define parameter labels
                param_labels = {'Temperature': 'Temp', 'Top-p': 'Topp', 'Top-k': 'Topk', 'BM25 Weight': 'BM25'}
                
                # Generate additional files
                self.generate_html_table(blocks, combined_df, self.color_mapping, outdir, param_labels)
                self.generate_docx_and_csv(blocks, combined_df, self.color_mapping, outdir, param_labels, file_to_varying_param)
                self.generate_stats_files(blocks, combined_df, self.color_mapping, outdir, param_labels)
            
        except Exception as e:
            print('Exception in process_merged_csvs:')
            import traceback
            traceback.print_exc()
            self.root.after(0, self.show_error, str(e))

    def process_single_csv_with_params(self):
        """Process a single CSV file that contains multiple parameter variations"""
        import re
        import traceback
        
        file_path = filedialog.askopenfilename(
            title="Select CSV file with parameter variations",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        # Store the file path and enable buttons
        self.csv_file = file_path
        self.file_label.config(text=os.path.basename(file_path), foreground="black")
        self.process_btn.config(state='normal')
        self.full_analysis_button.config(state='normal')
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path, encoding='utf-8')
            
            # Check if the file has the expected parameter columns
            expected_params = ['Temperature', 'Top-p', 'Top-k', 'BM25 Weight']
            missing_params = [param for param in expected_params if param not in df.columns]
            
            if missing_params:
                msg = f"CSV file is missing required parameter columns: {', '.join(missing_params)}"
                self.merge_label.config(text=msg, foreground="red")
                messagebox.showerror("Parameter Error", msg)
                return
            
            # Check for error patterns in the data
            error_patterns = [
                r"error generating response:",
                r"api error occurred:",
                r"bad gateway",
                r"cloudflare",
                r"server disconnected without sending a response",
                r"getaddrinfo failed"
            ]
            
            error_found = False
            error_msgs = []
            
            for idx, row in df.iterrows():
                for col in df.columns:
                    val = str(row[col]).lower()
                    for pat in error_patterns:
                        if re.search(pat, val):
                            error_found = True
                            error_msgs.append(f"Row: {idx+2}, Column: '{col}', Error: {row[col]}")
            
            if error_found:
                msg = "Processing aborted due to invalid data in the following locations:\n" + "\n".join(error_msgs)
                self.merge_label.config(text=msg, foreground="red")
                messagebox.showerror("Processing Error", msg)
                return
            
            # Extract concepts from the Main Answer column
            if 'Main Answer' not in df.columns:
                msg = "CSV file is missing 'Main Answer' column"
                self.merge_label.config(text=msg, foreground="red")
                messagebox.showerror("Column Error", msg)
                return
            
            # Extract concepts and quotes from each row
            concept_quote_results = df['Main Answer'].apply(self.extract_concepts_and_quotes)
            df['Concepts'] = [result[0] for result in concept_quote_results]
            df['Concept_Quotes'] = [result[1] for result in concept_quote_results]
            
            # Check dataset size to prevent memory issues
            total_rows = len(df)
            if total_rows > 1000:
                print(f"Warning: Large dataset detected ({total_rows} rows). This may cause memory issues.")
                response = messagebox.askyesno("Large Dataset Warning", 
                                             f"Dataset has {total_rows} rows which may cause memory issues.\n"
                                             "Consider processing a smaller subset or ensure you have sufficient RAM.\n\n"
                                             "Continue anyway?")
                if not response:
                    return
            
            # Create output directory
            output_dir = os.path.join(os.getcwd(), "single_csv_parameter_analysis")
            os.makedirs(output_dir, exist_ok=True)
            
            # Process each parameter separately for UpSet plots
            param_results = {}
            
            for i, param in enumerate(expected_params):
                print(f"Processing parameter {i+1}/{len(expected_params)}: {param}")
                self.merge_label.config(text=f"Processing parameter {i+1}/{len(expected_params)}: {param}", foreground="blue")
                self.root.update()  # Update the UI
                
                # Get unique values for this parameter
                unique_values = sorted(df[param].unique())
                print(f"Unique values for {param}: {unique_values}")
                
                # Create concept sets for each value of this parameter
                concept_sets = {}
                
                for value in unique_values:
                    # Get rows where this parameter has this value
                    param_rows = df[df[param] == value]
                    print(f"Processing {param}={value}: {len(param_rows)} rows")
                    
                    # Extract all concepts from these rows
                    all_concepts = []
                    for _, row in param_rows.iterrows():
                        concepts = row['Concepts']
                        if concepts:
                            all_concepts.extend(concepts)
                    
                    # Remove duplicates and create a set
                    concept_sets[f"{param}={value}"] = set(all_concepts)
                    print(f"  Found {len(set(all_concepts))} unique concepts")
                
                # Create UpSet plot for this parameter
                param_output_dir = os.path.join(output_dir, f"{param.replace(' ', '_').replace('-', '_')}_analysis")
                os.makedirs(param_output_dir, exist_ok=True)
                
                # Create DataFrame for UpSet
                all_concepts = set()
                for concepts in concept_sets.values():
                    all_concepts.update(concepts)
                
                # Create indicator matrix
                upset_df = pd.DataFrame(index=list(all_concepts))
                for set_name, concepts in concept_sets.items():
                    upset_df[set_name] = [concept in concepts for concept in all_concepts]
            
            # Generate UpSet plot
            try:
                from upsetplot import from_indicators, UpSet
                import matplotlib.pyplot as plt
                
                # Create a proper indicator DataFrame for UpSet
                # Convert to boolean and ensure proper structure
                upset_df = upset_df.astype(bool)
                
                # Only create UpSet if we have multiple sets and concepts
                if len(concept_sets) > 1 and len(all_concepts) > 0:
                    try:
                        # Ensure proper DataFrame structure for upsetplot
                        upset_df = upset_df.astype(bool)
                        upset_df = upset_df.reset_index(drop=True)  # Reset index to avoid index issues
                        
                        upset_data = from_indicators(upset_df, upset_df.columns)
                        
                        fig, axes = plt.subplots(1, 1, figsize=(12, 8))
                        upset = UpSet(upset_data, show_counts=True)
                        upset.plot(fig=fig)
                        
                        plt.title(f"UpSet Plot for {param}")
                        plt.tight_layout()
                        
                        plot_path = os.path.join(param_output_dir, f"upset_plot_{param.replace(' ', '_')}.png")
                        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        print(f"UpSet plot saved: {plot_path}")
                    except Exception as plot_error:
                        print(f"Error creating UpSet plot for {param}: {plot_error}")
                        print(f"DataFrame shape: {upset_df.shape}, columns: {list(upset_df.columns)}")
                        
                else:
                    print(f"Skipping UpSet plot for {param}: insufficient data (sets: {len(concept_sets)}, concepts: {len(all_concepts)})")
                
                param_results[param] = concept_sets
                
            except Exception as e:
                print(f"Error creating UpSet plot for {param}: {e}")
                param_results[param] = concept_sets
            
            # Create summary report
            summary_path = os.path.join(output_dir, "parameter_analysis_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("Parameter Analysis Summary\n")
                f.write("=" * 50 + "\n\n")
                
                for param, concept_sets in param_results.items():
                    f.write(f"{param}:\n")
                    f.write("-" * 20 + "\n")
                    for set_name, concepts in concept_sets.items():
                        f.write(f"  {set_name}: {len(concepts)} concepts\n")
                        if concepts:
                            sample_concepts = list(concepts)[:5]
                            safe_concepts = []
                            for concept in sample_concepts:
                                try:
                                    concept.encode('utf-8').decode('utf-8')
                                    safe_concepts.append(concept)
                                except UnicodeError:
                                    safe_concepts.append(concept.encode('ascii', 'replace').decode('ascii'))
                            f.write(f"    Sample: {', '.join(safe_concepts)}\n")
                    f.write("\n")
            
            print(f"Analysis complete. Results saved to: {output_dir}")
            self.merge_label.config(text=f"Analysis complete. Results saved to: {output_dir}", foreground="green")
            
            # Open the output directory
            import subprocess
            import platform
            if platform.system() == "Windows":
                subprocess.run(["explorer", output_dir])
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", output_dir])
            else:  # Linux
                subprocess.run(["xdg-open", output_dir])
                
        except Exception as e:
            error_msg = f"Error processing CSV: {str(e)}"
            print(error_msg)
            self.merge_label.config(text=f"Error: {str(e)}", foreground="red")
            messagebox.showerror("Processing Error", error_msg)

    def generate_full_analysis(self):
        """Generate full analysis with all files (HTML, DOCX, stats, etc.) for the selected single CSV"""
        if not hasattr(self, 'csv_file') or not self.csv_file:
            messagebox.showerror("Error", "No CSV file selected. Please use 'Select Single CSV' first.")
            return
            
        try:
            # Start progress
            self.progress.start()
            self.full_analysis_button.config(state='disabled')
            self.merge_label.config(text="Generating full analysis...", foreground="blue")
            self.root.update()
            
            # Read the CSV file
            df = pd.read_csv(self.csv_file, encoding='utf-8')
            
            # Check if the file has the expected parameter columns
            expected_params = ['Temperature', 'Top-p', 'Top-k', 'BM25 Weight']
            missing_params = [param for param in expected_params if param not in df.columns]
            
            if missing_params:
                msg = f"CSV file is missing required parameter columns: {', '.join(missing_params)}"
                self.merge_label.config(text=msg, foreground="red")
                messagebox.showerror("Parameter Error", msg)
                return
            
            # Extract concepts from the Main Answer column
            if 'Main Answer' not in df.columns:
                msg = "CSV file is missing 'Main Answer' column"
                self.merge_label.config(text=msg, foreground="red")
                messagebox.showerror("Column Error", msg)
                return
            
            # Extract concepts and quotes from each row
            concept_quote_results = df['Main Answer'].apply(self.extract_concepts_and_quotes)
            df['Concepts'] = [result[0] for result in concept_quote_results]
            df['Concept_Quotes'] = [result[1] for result in concept_quote_results]
            
            # Create color mapping
            all_concepts = set()
            for concepts in df['Concepts']:
                all_concepts.update(concepts)
            self.create_color_mapping(list(all_concepts))
            
            # Create output directory
            output_dir = os.path.join(os.getcwd(), "single_csv_full_analysis")
            os.makedirs(output_dir, exist_ok=True)
            
            # Set up parameters and labels
            param_labels = {'Temperature': 'Temp', 'Top-p': 'Topp', 'Top-k': 'Topk', 'BM25 Weight': 'BM25'}
            
            # Get detected blocks or create default blocks
            blocks = getattr(self, 'detected_blocks', None)
            if not blocks or len(blocks) == 0:
                # Create default blocks for 5x5x5x5 structure
                n = len(df)
                if n == 625:  # 5x5x5x5
                    block_size = 125  # 5^3
                    blocks = [(i*block_size, (i+1)*block_size, expected_params[i] if i < len(expected_params) else '?') for i in range(4)]
                else:
                    block_size = n // 4 if n % 4 == 0 else 5
                    blocks = [(i*block_size, (i+1)*block_size, expected_params[i] if i < len(expected_params) else '?') for i in range(4)]
            
            # Generate all files like the original process_data function
            output_files = []
            
            # Generate UpSet plots for each block
            for start, end, varying_param in blocks:
                print(f'Processing block: {varying_param} ({start}-{end})')
                subset = df.iloc[start:end].copy()
                if subset.empty:
                    print('Subset empty, skipping')
                    continue
                
                # Create concept matrix
                from sklearn.preprocessing import MultiLabelBinarizer
                mlb = MultiLabelBinarizer()
                concept_matrix = pd.DataFrame(mlb.fit_transform(subset['Concepts']), columns=mlb.classes_)
                
                # Add parameter columns
                for p in expected_params:
                    concept_matrix[p] = subset[p].values
                
                concept_matrix_reset = concept_matrix.drop(expected_params, axis=1).astype(bool).reset_index(drop=True)
                
                # Apply color grouping if enabled
                if self.use_colors.get() and self.group_by_same_color.get():
                    from collections import defaultdict
                    color_map = {col: self.color_mapping.get(col, None) for col in concept_matrix_reset.columns}
                    color_groups = defaultdict(list)
                    for col, color in color_map.items():
                        color_groups[color].append(col)
                    
                    merged = pd.DataFrame(index=concept_matrix_reset.index)
                    for color, cols in color_groups.items():
                        if color is None or len(cols) == 0:
                            continue
                        if len(cols) == 1:
                            merged[cols[0]] = concept_matrix_reset[cols[0]]
                        else:
                            group_name = "/".join(cols)
                            group_name_wrapped = wrap_label(group_name, width=self.get_wrap_width())
                            merged[group_name_wrapped] = concept_matrix_reset[cols].any(axis=1)
                    concept_matrix_reset = merged
                
                # Create UpSet plot
                from upsetplot import from_indicators, UpSet
                import matplotlib.pyplot as plt
                import numpy as np
                
                upset_data = from_indicators(concept_matrix_reset, concept_matrix_reset.columns)
                fig = plt.figure(figsize=(12, 8))
                upset = UpSet(upset_data, show_counts=True)
                axes = upset.plot(fig=fig)
                bar_ax = axes['intersections']
                matrix_ax = axes['matrix']
                bars = bar_ax.patches
                upset_index = upset_data.index
                
                # Apply color and label logic
                wrap_width = self.get_wrap_width()
                if self.use_colors.get():
                    yticks = matrix_ax.get_yticklabels()
                    wrapped_labels = []
                    for label in yticks:
                        concept = label.get_text()
                        wrapped = wrap_label(concept, width=wrap_width)
                        wrapped_labels.append(wrapped)
                    matrix_ax.set_yticklabels(wrapped_labels)
                    for label, concept in zip(matrix_ax.get_yticklabels(), [l.get_text().replace('\n', ' ') for l in yticks]):
                        if concept in self.color_mapping:
                            color = self.color_mapping[concept]
                        else:
                            first_concept = concept.split('/')[0]
                            color = self.color_mapping.get(first_concept, 'black')
                        label.set_color(color)
                        label.set_weight('bold')
                        label.set_fontsize(10)
                else:
                    yticks = matrix_ax.get_yticklabels()
                    wrapped_labels = [wrap_label(label.get_text(), width=wrap_width) for label in yticks]
                    matrix_ax.set_yticklabels(wrapped_labels)
                    for label in matrix_ax.get_yticklabels():
                        label.set_color('black')
                        label.set_weight('normal')
                        label.set_fontsize(10)
                
                # Add parameter value labels - Improved logic with memory management
                significant_bars = []
                for i, (bar, intersection) in enumerate(zip(bars, upset_index)):
                    if i == 0:  # Skip the empty intersection
                        continue
                    if bar.get_height() > 0:
                        significant_bars.append((i, bar, intersection))
                
                # Reduce number of labels to prevent memory issues and improve readability
                max_labels = min(10, len(significant_bars))  # Reduced from 20 to 10
                if len(significant_bars) > max_labels:
                    significant_bars.sort(key=lambda x: x[1].get_height(), reverse=True)
                    significant_bars = significant_bars[:max_labels]
                
                # Create a mapping from intersection patterns to row indices
                intersection_to_rows = {}
                
                # For each row in the subset, determine which intersection it belongs to
                for row_idx in range(len(subset)):
                    row_concepts = subset.iloc[row_idx]['Concepts']
                    row_pattern = []
                    
                    # Check which concepts are present for this row
                    for col in concept_matrix_reset.columns:
                        # Handle grouped concepts (split by '/')
                        original_cols = col.split('/') if '/' in col else [col]
                        concept_present = any(concept in row_concepts for concept in original_cols)
                        row_pattern.append(concept_present)
                    
                    # Convert pattern to tuple for hashing
                    pattern_tuple = tuple(row_pattern)
                    if pattern_tuple not in intersection_to_rows:
                        intersection_to_rows[pattern_tuple] = []
                    intersection_to_rows[pattern_tuple].append(row_idx)
                
                # Track labels to avoid duplicates and reduce clutter
                drawn_labels = set()
                
                for i, bar, intersection in significant_bars:
                    x = bar.get_x() + bar.get_width() / 2 + 1  # Skip first column by adding offset
                    
                    # Find which rows contribute to this intersection
                    contributing_rows = intersection_to_rows.get(intersection, [])
                    
                    if contributing_rows:
                        # Get parameter values from contributing rows
                        param_vals = subset.iloc[contributing_rows][varying_param].unique()
                        
                        def fmt(v):
                            try:
                                f = float(v)
                                return f"{f:.1f}"
                            except Exception:
                                return str(v)
                        
                        label = ','.join(fmt(v) for v in param_vals) if len(param_vals) > 0 else ''
                        
                        # Only draw label if it's not a duplicate and not too long
                        if label and label not in drawn_labels and len(label) <= 20:
                            print(f"Drawing parameter label: '{label}' at x={x} (rows: {contributing_rows})")
                            # Position the label aligned with the top of the first row
                            y_label = len(concept_matrix_reset.columns) - 0.3  # Position at top of first row
                            matrix_ax.text(x, y_label, label, ha='center', va='bottom', fontsize=8, color='red', rotation=90, clip_on=False, weight='bold')
                            drawn_labels.add(label)
                    else:
                        print(f"No contributing rows found for intersection {i} at x={x}")
                
                # Add title and subtitle
                other_params = [p for p in expected_params if p != varying_param]
                fixed_vals = {param_labels[p]: subset[p].iloc[0] for p in other_params}
                fixed_str = ' | '.join(f"{p}={v}" for p, v in fixed_vals.items())
                plt.suptitle(f"UpSet Diagram: {param_labels[varying_param]} sweep\nOther Params: {fixed_str}", fontsize=14, y=0.98)
                plt.subplots_adjust(top=0.88, bottom=0.12)
                plt.tight_layout(rect=[0, 0.12, 1, 0.88])
                
                outpath = f"{output_dir}/compare_{param_labels[varying_param]}_composed.png"
                try:
                    plt.savefig(outpath, dpi=150, bbox_inches='tight', pad_inches=0.5, format='png')
                except Exception as e:
                    print(f"Error saving PNG diagram: {e}")
                plt.close('all')
                
                output_files.append(outpath)
                
                # Clean up memory
                import gc
                del concept_matrix, concept_matrix_reset, upset_data, upset, fig, axes, bars, upset_index
                gc.collect()
            
            # Generate additional files
            self.generate_html_table(blocks, df, self.color_mapping, output_dir, param_labels)
            self.generate_docx_and_csv(blocks, df, self.color_mapping, output_dir, param_labels)
            self.generate_stats_files(blocks, df, self.color_mapping, output_dir, param_labels)
            
            # Update status (don't call update_results to avoid memory issues)
            self.merge_label.config(text=f"Full analysis complete. Generated {len(output_files)} plots and additional files. Results saved to: {output_dir}", foreground="green")
            
            # Open the output directory
            import subprocess
            import platform
            if platform.system() == "Windows":
                subprocess.run(["explorer", output_dir])
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", output_dir])
            else:  # Linux
                subprocess.run(["xdg-open", output_dir])
                
        except Exception as e:
            error_msg = f"Error generating full analysis: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.merge_label.config(text=f"Error: {str(e)}", foreground="red")
            messagebox.showerror("Processing Error", error_msg)
        finally:
            # Stop progress and re-enable button
            self.progress.stop()
            self.full_analysis_button.config(state='normal')

    def generate_parameter_plots(self, df, blocks, color_mapping, output_dir, param_name):
        """Generate plots for a specific parameter variation"""
        try:
            import matplotlib.pyplot as plt
            from upsetplot import UpSet, from_indicators
            import pandas as pd
            import gc  # For garbage collection
            
            # Check if sklearn is available
            try:
                from sklearn.preprocessing import MultiLabelBinarizer
                sklearn_available = True
            except ImportError:
                print("Warning: sklearn not available, using fallback method for concept matrix")
                sklearn_available = False
            
            # Create concept matrix for UpSet plot
            all_concepts = []
            for concepts in df['Concepts']:
                all_concepts.extend(concepts)
            all_concepts = list(set(all_concepts))
            
            # Create concept matrix - use MultiLabelBinarizer for proper format
            if sklearn_available:
                mlb = MultiLabelBinarizer()
                concept_matrix = mlb.fit_transform(df['Concepts'])
                concept_df = pd.DataFrame(concept_matrix, columns=mlb.classes_)
            else:
                # Fallback method without sklearn
                concept_matrix = []
                for idx, row in df.iterrows():
                    row_concepts = set(row['Concepts'])
                    concept_row = [1 if concept in row_concepts else 0 for concept in all_concepts]
                    concept_matrix.append(concept_row)
                concept_df = pd.DataFrame(concept_matrix, columns=all_concepts)
            
            # Add parameter value as index
            param_values = df[param_name].values
            concept_df.index = [f"{param_name}={val}" for val in param_values]
            
            # Create UpSet plot - only if we have data
            if len(concept_df) > 0 and len(concept_df.columns) > 0:
                try:
                    # Use the correct format for upsetplot
                    concept_df_bool = concept_df.astype(bool)
                    upset_data = from_indicators(concept_df_bool)
                    upset = UpSet(upset_data, min_subset_size=1, show_counts=True)
                    fig = plt.figure(figsize=(12, 8))
                    upset.plot()
                    plt.title(f"Concept Overlaps by {param_name} Values")
                    plt.tight_layout()
                except Exception as upset_error:
                    print(f"Error creating UpSet plot for {param_name}: {upset_error}")
                    # Skip UpSet plot if it fails
                    fig = None
            else:
                print(f"No valid data for UpSet plot in {param_name}")
                fig = None
            
            # Save plot
            if fig is not None:
                plot_path = os.path.join(output_dir, f"{param_name.replace(' ', '_').replace('-', '_')}_upset_plot.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                del fig
                gc.collect()  # Force garbage collection
            else:
                print(f"Skipping UpSet plot save for {param_name} - no valid figure")
            
            # Create parameter value vs concept count plot
            concept_counts = [len(concepts) for concepts in df['Concepts']]
            param_values = df[param_name].values
            
            fig2 = plt.figure(figsize=(10, 6))
            plt.scatter(param_values, concept_counts, alpha=0.7, s=50)
            plt.xlabel(param_name)
            plt.ylabel('Number of Concepts')
            plt.title(f'Concept Count vs {param_name}')
            plt.grid(True, alpha=0.3)
            
            # Add trend line
            if len(param_values) > 1:
                z = np.polyfit(param_values, concept_counts, 1)
                p = np.poly1d(z)
                plt.plot(param_values, p(param_values), "r--", alpha=0.8)
            
            plot_path = os.path.join(output_dir, f"{param_name.replace(' ', '_').replace('-', '_')}_concept_count.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig2)
            del fig2
            gc.collect()  # Force garbage collection
            
            # Create heatmap of concept frequency by parameter value
            concept_freq_matrix = []
            unique_param_values = sorted(df[param_name].unique())
            
            for param_val in unique_param_values:
                param_rows = df[df[param_name] == param_val]
                concept_freq = {}
                for concepts in param_rows['Concepts']:
                    for concept in concepts:
                        concept_freq[concept] = concept_freq.get(concept, 0) + 1
                
                freq_row = [concept_freq.get(concept, 0) for concept in all_concepts]
                concept_freq_matrix.append(freq_row)
            
            if concept_freq_matrix:
                freq_df = pd.DataFrame(concept_freq_matrix, 
                                     index=[f"{param_name}={val}" for val in unique_param_values],
                                     columns=all_concepts)
                
                # Create heatmap using matplotlib if seaborn is not available
                if sns is not None:
                    fig3 = plt.figure(figsize=(max(12, len(all_concepts) * 0.3), 8))
                    sns.heatmap(freq_df.T, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Frequency'})
                    plt.title(f'Concept Frequency by {param_name}')
                    plt.xlabel(param_name)
                    plt.ylabel('Concepts')
                    plt.xticks(rotation=45)
                    plt.yticks(rotation=0)
                    plt.tight_layout()
                else:
                    # Fallback to matplotlib heatmap
                    fig3 = plt.figure(figsize=(max(12, len(all_concepts) * 0.3), 8))
                    plt.imshow(freq_df.T.values, cmap='YlOrRd', aspect='auto')
                    plt.colorbar(label='Frequency')
                    plt.title(f'Concept Frequency by {param_name}')
                    plt.xlabel(param_name)
                    plt.ylabel('Concepts')
                    plt.xticks(range(len(unique_param_values)), [f"{param_name}={val}" for val in unique_param_values], rotation=45)
                    plt.yticks(range(len(all_concepts)), all_concepts)
                    
                    # Add text annotations - limit to prevent memory issues
                    max_annotations = 100  # Limit annotations to prevent memory issues
                    annotation_count = 0
                    for i in range(len(all_concepts)):
                        for j in range(len(unique_param_values)):
                            if annotation_count >= max_annotations:
                                break
                            plt.text(j, i, str(freq_df.T.values[i, j]), ha='center', va='center')
                            annotation_count += 1
                        if annotation_count >= max_annotations:
                            break
                    
                    plt.tight_layout()
                
                plot_path = os.path.join(output_dir, f"{param_name.replace(' ', '_').replace('-', '_')}_heatmap.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig3)
                del fig3
                gc.collect()  # Force garbage collection
            
            # Clean up variables to free memory
            del concept_matrix, concept_df
            if 'freq_df' in locals():
                del freq_df
            gc.collect()
            
        except Exception as e:
            print(f"Error generating plots for {param_name}: {e}")
            import traceback
            traceback.print_exc()
            # Clean up any remaining matplotlib objects
            plt.close('all')
            gc.collect()

    def create_parameter_summary_report(self, param_results, output_dir):
        """Create a summary report of all parameter analyses"""
        try:
            import docx
            from docx.shared import RGBColor
            
            doc = docx.Document()
            doc.add_heading('Parameter Variation Analysis Summary', 0)
            
            # Add summary table
            table = doc.add_table(rows=1, cols=4)
            hdr_cells = table.rows[0].cells
            hdr_cells[0].text = 'Parameter'
            hdr_cells[1].text = 'Unique Values'
            hdr_cells[2].text = 'Value Range'
            hdr_cells[3].text = 'Output Directory'
            
            for param_name, results in param_results.items():
                row_cells = table.add_row().cells
                row_cells[0].text = param_name
                row_cells[1].text = str(len(results['unique_values']))
                row_cells[2].text = f"{min(results['unique_values'])} - {max(results['unique_values'])}"
                row_cells[3].text = os.path.basename(results['output_dir'])
            
            # Add detailed analysis for each parameter
            for param_name, results in param_results.items():
                doc.add_heading(f'{param_name} Analysis', level=1)
                
                # Parameter statistics
                doc.add_paragraph(f"Parameter: {param_name}")
                doc.add_paragraph(f"Number of unique values: {len(results['unique_values'])}")
                doc.add_paragraph(f"Value range: {min(results['unique_values'])} - {max(results['unique_values'])}")
                
                # Concept statistics by parameter value
                doc.add_heading('Concept Statistics by Parameter Value', level=2)
                
                stats_table = doc.add_table(rows=1, cols=3)
                stats_hdr = stats_table.rows[0].cells
                stats_hdr[0].text = f'{param_name} Value'
                stats_hdr[1].text = 'Number of Rows'
                stats_hdr[2].text = 'Average Concepts per Row'
                
                for value in results['unique_values']:
                    # Skip detailed analysis if DataFrame is not available (to save memory)
                    row_cells = stats_table.add_row().cells
                    row_cells[0].text = str(value)
                    row_cells[1].text = "N/A"  # We don't have the DataFrame anymore
                    row_cells[2].text = "N/A"  # We don't have the DataFrame anymore
                
                doc.add_paragraph()  # Add spacing
            
            # Save the report
            report_path = os.path.join(output_dir, 'parameter_analysis_summary.docx')
            doc.save(report_path)
            
        except Exception as e:
            print(f"Error creating summary report: {e}")
            import traceback
            traceback.print_exc()
            # Create a simple text report as fallback
            try:
                report_path = os.path.join(output_dir, 'parameter_analysis_summary.txt')
                with open(report_path, 'w') as f:
                    f.write("Parameter Variation Analysis Summary\n")
                    f.write("=" * 40 + "\n\n")
                    for param_name, results in param_results.items():
                        f.write(f"Parameter: {param_name}\n")
                        f.write(f"Unique values: {len(results['unique_values'])}\n")
                        f.write(f"Value range: {min(results['unique_values'])} - {max(results['unique_values'])}\n")
                        f.write(f"Output directory: {os.path.basename(results['output_dir'])}\n\n")
                print(f"Created fallback text report: {report_path}")
            except Exception as fallback_error:
                print(f"Error creating fallback report: {fallback_error}")

    def generate_docx_and_csv(self, blocks, df, color_mapping, outdir, param_labels, file_to_varying_param=None):
        try:
            import docx
            from docx.shared import RGBColor
            import csv
            doc = docx.Document()
            table = doc.add_table(rows=1, cols=5)
            hdr_cells = table.rows[0].cells
            hdr_cells[0].text = 'Temperature'
            hdr_cells[1].text = 'Top P'
            hdr_cells[2].text = 'Top K'
            hdr_cells[3].text = 'BM25 Weight'
            hdr_cells[4].text = 'Extracted Concepts'
            for start, end, varying_param in blocks:
                subset = df.iloc[start:end].copy()
                for idx, row in subset.iterrows():
                    row_cells = table.add_row().cells
                    row_cells[0].text = str(row.get('Temperature',''))
                    row_cells[1].text = str(row.get('Top-p',''))
                    row_cells[2].text = str(row.get('Top-k',''))
                    row_cells[3].text = str(row.get('BM25 Weight',''))
                    concept_strs = []
                    for concept in row['Concepts']:
                        color = color_mapping.get(concept, '#888')
                        concept_strs.append(f"{concept} [{color}]")
                    row_cells[4].text = ", ".join(concept_strs)
            docx_path = os.path.join(outdir, "compare.docx")
            doc.save(docx_path)
            csv_path = os.path.join(outdir, "compare.csv")
            with open(csv_path, "w", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(['Temperature', 'Top P', 'Top K', 'BM25 Weight', 'Extracted Concepts'])
                for start, end, varying_param in blocks:
                    subset = df.iloc[start:end].copy()
                    for idx, row in subset.iterrows():
                        concept_strs = []
                        for concept in row['Concepts']:
                            color = color_mapping.get(concept, '#888')
                            concept_strs.append(f"{concept} [{color}]")
                        writer.writerow([
                            row.get('Temperature',''),
                            row.get('Top-p',''),
                            row.get('Top-k',''),
                            row.get('BM25 Weight',''),
                            ", ".join(concept_strs)
                        ])
        except Exception as e:
            print(f"Error generating DOCX/CSV: {e}")

    def generate_stats_files(self, blocks, df, color_mapping, outdir, param_labels):
        try:
            import pandas as pd
            import docx
            from docx.shared import RGBColor
            from docx.enum.text import WD_ALIGN_PARAGRAPH
            from docx.enum.section import WD_ORIENT
            from docx2pdf import convert as docx2pdf_convert
            import os
            txt_lines = []
            all_concepts = sorted(set(c for concepts in df['Concepts'] for c in concepts))
            # TXT
            txt_lines.append("Concepts Overview\n================\n")
            txt_lines.append(f"Total unique concepts: {len(all_concepts)}\n")
            txt_lines.append("\nAll Concepts:\n" + "\n".join(all_concepts) + "\n")
            txt_lines.append("\nConcepts per block:\n")
            for start, end, varying_param in blocks:
                subset = df.iloc[start:end].copy()
                block_concepts = sorted(set(c for concepts in subset['Concepts'] for c in concepts))
                txt_lines.append(f"{varying_param} ({start}-{end-1}): {len(block_concepts)} concepts\n" + ", ".join(block_concepts) + "\n")
            txt_lines.append("\nColor Groups:\n")
            color_groups = {}
            for concept, color in color_mapping.items():
                color_groups.setdefault(color, []).append(concept)
            for color, concepts in color_groups.items():
                txt_lines.append(f"Color {color}: {len(concepts)} concepts\n" + ", ".join(concepts) + "\n")
            txt_path = os.path.join(outdir, "compare_stats.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("".join(txt_lines))
            # DOCX
            doc = docx.Document()
            section = doc.sections[0]
            from docx.shared import Inches
            from docx.enum.section import WD_ORIENT
            section.orientation = WD_ORIENT.LANDSCAPE
            # Swap width and height for landscape
            new_width, new_height = section.page_height, section.page_width
            section.page_width = new_width
            section.page_height = new_height
            # Set narrow margins and header/footer
            section.top_margin = Inches(0.3)
            section.bottom_margin = Inches(0.3)
            section.left_margin = Inches(0.3)
            section.right_margin = Inches(0.3)
            section.header_distance = Inches(0.3)
            section.footer_distance = Inches(0.3)
            # --- First page: folder name as title ---
            folder_name = getattr(self, 'merged_folder_name', '')
            if folder_name:
                doc.add_heading(folder_name, level=1)
            else:
                doc.add_heading("Merged Results", level=1)
            # Insert 4 images in a table, side by side, width matches table width
            image_dir = os.path.join(os.path.dirname(outdir), "compare_gui_output")
            image_files = [
                "compare_BM25_composed.png",
                "compare_Topk_composed.png",
                "compare_Topp_composed.png",
                "compare_Temp_composed.png"
            ]
            img_table = doc.add_table(rows=1, cols=4)
            img_table.autofit = False
            # Calculate available width and height for images (page width - margins)
            table_width = int(section.page_width - section.left_margin - section.right_margin)
            available_height = int(section.page_height - section.top_margin - section.bottom_margin)
            img_width = int(table_width // 4)
            from PIL import Image as PILImage
            for i, img_file in enumerate(image_files):
                img_path = os.path.join(image_dir, img_file)
                if os.path.exists(img_path):
                    cell = img_table.rows[0].cells[i]
                    cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                    run = cell.paragraphs[0].add_run()
                    # Open image to get aspect ratio
                    with PILImage.open(img_path) as pil_img:
                        aspect = pil_img.width / pil_img.height
                        # Calculate height in inches to fill page height
                        height_in_inches = (available_height / 914400)
                        height_in_inches = max(0.5, height_in_inches - 0.1)
                        # Calculate width to maintain aspect ratio, but cap at img_width
                        width_in_inches = min(img_width / 914400, aspect * height_in_inches)
                        run.add_picture(img_path, width=docx.shared.Inches(width_in_inches))
            # --- Page break, then Color Groups section ---
            doc.add_heading("Color Groups", level=1)
            # Color Groups table: 4 columns, set widths proportionally to fill table_width
            color_table = doc.add_table(rows=1, cols=4)
            color_table.autofit = False
            col_props = [1.2, 0.4, 2.2, 3.2]
            total = sum(col_props)
            col_widths = [int(table_width * (w / total)) for w in col_props]
            for i, w in enumerate(col_widths):
                color_table.columns[i].width = w
            color_table.rows[0].cells[0].text = 'Group Label'
            color_table.rows[0].cells[1].text = 'Count'
            color_table.rows[0].cells[2].text = 'Concepts'
            color_table.rows[0].cells[3].text = 'Parameters'
            for color, concepts in color_groups.items():
                # Group label: common word in group (use the longest common substring or first word)
                group_label = ''
                if len(concepts) > 1:
                    from difflib import SequenceMatcher
                    def lcs(a, b):
                        match = SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
                        return a[match.a: match.a + match.size]
                    lcs_str = concepts[0]
                    for c in concepts[1:]:
                        lcs_str = lcs(lcs_str, c)
                    group_label = lcs_str.strip() if lcs_str.strip() else concepts[0].split()[0]
                else:
                    group_label = concepts[0].split()[0]
                row_cells = color_table.add_row().cells
                row_cells[0].text = group_label
                row_cells[1].text = str(len(concepts))
                # Concepts (colored)
                para = row_cells[2].paragraphs[0]
                for i, concept in enumerate(concepts):
                    run = para.add_run(concept)
                    if color.startswith('#') and len(color) == 7:
                        r, g, b = tuple(int(color[j:j+2], 16) for j in (1, 3, 5))
                        run.font.color.rgb = RGBColor(r, g, b)
                    if i < len(concepts) - 1:
                        para.add_run(", ")
                # Parameters (expanded cell)
                param_strs = []
                for concept in concepts:
                    param_rows = df[df['Concepts'].apply(lambda lst: concept in lst)]
                    for _, row in param_rows.iterrows():
                        param_strs.append(f"(Temp: {row.get('Temperature','')}, Topp: {row.get('Top-p','')}, Topk: {row.get('Top-k','')}, BM25: {row.get('BM25 Weight','')})")
                row_cells[3].text = ", ".join(param_strs)
                # Color group label cell colored
                for para in row_cells[0].paragraphs:
                    for run in para.runs:
                        if color.startswith('#') and len(color) == 7:
                            r, g, b = tuple(int(color[j:j+2], 16) for j in (1, 3, 5))
                            run.font.color.rgb = RGBColor(r, g, b)
            # --- All Concepts Table ---
            doc.add_heading("All Concepts", level=1)
            table = doc.add_table(rows=1, cols=2)
            table.autofit = True
            hdr_cells = table.rows[0].cells
            hdr_cells[0].text = 'Index'
            hdr_cells[1].text = 'Concept'
            for idx, concept in enumerate(all_concepts):
                row_cells = table.add_row().cells
                row_cells[0].text = str(idx+1)
                para = row_cells[1].paragraphs[0]
                run = para.add_run(concept)
                color = color_mapping.get(concept, '#888')
                if color.startswith('#') and len(color) == 7:
                    r, g, b = tuple(int(color[j:j+2], 16) for j in (1, 3, 5))
                    run.font.color.rgb = RGBColor(r, g, b)
            # --- Concepts per Block Table ---
            doc.add_heading("Concepts per Block", level=1)
            block_table = doc.add_table(rows=1, cols=3)
            block_table.autofit = True
            block_table.rows[0].cells[0].text = 'Block'
            block_table.rows[0].cells[1].text = 'Concepts'
            block_table.rows[0].cells[2].text = 'Count'
            for start, end, varying_param in blocks:
                subset = df.iloc[start:end].copy()
                block_concepts = sorted(set(c for concepts in subset['Concepts'] for c in concepts))
                row_cells = block_table.add_row().cells
                row_cells[0].text = varying_param
                para = row_cells[1].paragraphs[0]
                for i, concept in enumerate(block_concepts):
                    run = para.add_run(concept)
                    color = color_mapping.get(concept, '#888')
                    if color.startswith('#') and len(color) == 7:
                        r, g, b = tuple(int(color[j:j+2], 16) for j in (1, 3, 5))
                        run.font.color.rgb = RGBColor(r, g, b)
                    if i < len(block_concepts) - 1:
                        para.add_run(", ")
                row_cells[2].text = str(len(block_concepts))
            # --- Concept Quotes Section ---
            self._add_quotes_section_to_stats_doc(doc, df, outdir, blocks, color_mapping)
            
            # --- UpSet Histogram Section ---
            doc.add_heading("UpSet Diagram Histograms", level=1)
            # For each block, add a table with parameter, value, and histogram (bar heights/counts)
            for start, end, varying_param in blocks:
                subset = df.iloc[start:end].copy()
                doc.add_heading(f"{varying_param} Sweep", level=2)
                hist_table = doc.add_table(rows=1, cols=3)
                hist_table.autofit = True
                hist_table.rows[0].cells[0].text = 'Parameter'
                hist_table.rows[0].cells[1].text = 'Value'
                hist_table.rows[0].cells[2].text = 'Histogram (bar counts)'
                # For each unique value of the varying parameter, count the number of concepts present
                for val in sorted(subset[varying_param].unique()):
                    mask = subset[varying_param] == val
                    present_concepts = [c for concepts in subset[mask]['Concepts'] for c in concepts]
                    hist_row = hist_table.add_row().cells
                    hist_row[0].text = varying_param
                    hist_row[1].text = str(val)
                    hist_row[2].text = str(len(present_concepts))
            docx_path = os.path.join(outdir, "compare_stats.docx")
            doc.save(docx_path)
            # PDF: convert DOCX to PDF
            LIBREOFFICE_PATH = r"C:\\Program Files\\LibreOffice\\program\\soffice.exe"
            result = subprocess.run([
                LIBREOFFICE_PATH,
                "--headless",
                "--convert-to", "pdf",
                docx_path,
                "--outdir", outdir
            ], check=True, capture_output=True, text=True)
            pdf_path = os.path.join(outdir, os.path.splitext(os.path.basename(docx_path))[0] + ".pdf")
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF not found at {pdf_path}")
            self.merge_label.config(text="Stats PDF generated successfully.", foreground="black")
        except Exception as e:
            print(f"Error generating stats TXT/PDF/DOCX: {e}")

    def _add_quotes_section_to_stats_doc(self, doc, df, outdir, blocks, color_mapping=None, llm_group_tuples=None, color_to_concepts=None):
        """Add concept quotes section to the stats document"""
        import time
        start_time = time.time()
        
        try:
            from docx.shared import Inches
            from docx.enum.text import WD_ALIGN_PARAGRAPH
            
            # Add header for quotes section
            doc.add_heading("Concept Citations and Quotes", level=1)
            doc.add_paragraph(
                "This table shows specific citations and quotes from the original texts for each concept, "
                "extracted from the parameter-specific CSV files. The quotes demonstrate how each concept "
                "is used in context within the philosophical texts."
            )
            
            # Get all unique concepts
            all_concepts = sorted(set(c for concepts in df['Concepts'] for c in concepts))
            
            # Create file-to-varying-parameter mapping from blocks
            file_to_varying_param = {}
            for start, end, varying_param in blocks:
                subset = df.iloc[start:end].copy()
                for _, row in subset.iterrows():
                    file_name = row.get('File', 'Unknown')
                    file_to_varying_param[file_name] = varying_param
                    print(f"[QUOTES DEBUG] File '{file_name}' has varying param: {varying_param}")
            
            print(f"[QUOTES DEBUG] File to varying param mapping: {file_to_varying_param}")
            
            # Create quotes table
            quotes_table = doc.add_table(rows=1, cols=5)  # Concept, Short Quote, Full Citation, Sources, Reason columns
            quotes_table.style = 'Table Grid'
            quotes_table.autofit = False
            
            # Calculate available table width (page width - margins)
            section = doc.sections[0]
            table_width = int(section.page_width - section.left_margin - section.right_margin)
            
            # Set proportional column widths to fit within page width (reduced last column)
            col_props = [1.2, 1.8, 2.2, 1.5, 2.3]  # Reduced last column from 2.8 to 2.3
            total = sum(col_props)
            col_widths = [int(table_width * (w / total)) for w in col_props]
            for i, w in enumerate(col_widths):
                quotes_table.columns[i].width = w
            
            # Header row
            header_cells = quotes_table.rows[0].cells
            header_cells[0].text = "Concept"
            header_cells[1].text = "Short Quote"
            header_cells[2].text = "Full Citation"
            header_cells[3].text = "Sources"
            header_cells[4].text = "Reason"
            
            # Make header bold
            for cell in quotes_table.rows[0].cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        run.bold = True
            
            # Group concepts by their color mapping (same as Unique/Common Concepts table)
            concept_groups = {}
            group_colors = {}
            
            if color_mapping:
                for concept in all_concepts:
                    if concept in color_mapping:
                        color = color_mapping[concept]
                        if color not in concept_groups:
                            concept_groups[color] = []
                            group_colors[color] = color
                        concept_groups[color].append(concept)
            
            # If no color mapping available, create single-concept groups
            if not concept_groups:
                for concept in all_concepts:
                    concept_groups[f"group_{concept}"] = [concept]
                    group_colors[f"group_{concept}"] = "#000000"
            
            print(f"[QUOTES DEBUG] Grouped {len(all_concepts)} concepts into {len(concept_groups)} groups")
            print(f"[QUOTES DEBUG] Color mapping available: {bool(color_mapping)}")
            if color_mapping:
                print(f"[QUOTES DEBUG] Sample color mapping: {dict(list(color_mapping.items())[:3])}")
            
            # Add data rows for each group
            for color, group_concepts in concept_groups.items():
                row = quotes_table.add_row()
                
                # Set group name as concept text with color
                concept_cell = row.cells[0]
                concept_para = concept_cell.paragraphs[0]
                
                # Create group name from concepts (show all concepts in the group)
                if len(group_concepts) == 1:
                    group_name = group_concepts[0]
                else:
                    group_name = ", ".join(group_concepts)
                
                concept_run = concept_para.add_run(group_name)
                
                # Apply color from color mapping (same as Unique/Common Concepts table)
                if color.startswith('#') and len(color) == 7:
                    from docx.shared import RGBColor
                    r, g, b = tuple(int(color[j:j+2], 16) for j in (1, 3, 5))
                    concept_run.font.color.rgb = RGBColor(r, g, b)
                
                # Look for quotes and additional info for all concepts in the group
                all_quotes = []
                all_sources = []
                all_reasons = []
                concept_quotes_map = {}  # Map concept names to their quotes
                
                for concept in group_concepts:
                    quote_info = self._get_detailed_quotes_for_concept(concept, df, file_to_varying_param)
                    print(f"[QUOTES DEBUG] Concept '{concept}' - quotes: {len(quote_info['quotes'])}, sources: {len(quote_info['sources'])}")
                    
                    if quote_info['quotes']:
                        # Filter out invalid quotes (numbers, empty strings, etc.)
                        valid_quotes = []
                        for quote in quote_info['quotes']:
                            if (isinstance(quote, str) and 
                                quote.strip() and 
                                not quote.strip().isdigit() and 
                                len(quote.strip()) > 10):  # Minimum quote length
                                valid_quotes.append(quote.strip())
                        
                        if valid_quotes:
                            all_quotes.extend(valid_quotes)
                            concept_quotes_map[concept] = valid_quotes
                            print(f"[QUOTES DEBUG] Added {len(valid_quotes)} valid quotes for '{concept}'")
                        else:
                            print(f"[QUOTES DEBUG] No valid quotes found for '{concept}'")
                    
                    if quote_info['sources']:
                        all_sources.extend(quote_info['sources'])
                    if quote_info['reason_text']:
                        all_reasons.append(quote_info['reason_text'])
                
                if all_quotes:
                    # Short Quote column (first quote with concept name, truncated)
                    first_concept = list(concept_quotes_map.keys())[0]
                    first_quote = concept_quotes_map[first_concept][0]
                    short_quote = f"{first_concept}: {first_quote}"
                    if len(short_quote) > 150:
                        short_quote = short_quote[:150] + "..."
                    row.cells[1].text = short_quote
                    
                    # Full Citation column (quotes with concept names)
                    formatted_quotes = []
                    for concept, quotes in concept_quotes_map.items():
                        for quote in quotes:
                            formatted_quotes.append(f"{concept}: {quote}")
                    
                    # Create a combined quote_info structure for formatting
                    combined_quote_info = {
                        'quotes': formatted_quotes,
                        'sources': all_sources,
                        'reason_text': '; '.join(all_reasons) if all_reasons else ''
                    }
                    full_citation = self._format_quotes_only(combined_quote_info)
                    row.cells[2].text = full_citation
                    
                    # Sources column (parameter details with per-source varying parameter)
                    self._format_sources_cell_with_per_source_varying_param(row.cells[3], combined_quote_info)
                    
                    # Reason column
                    reason_text = self._format_reason_text(combined_quote_info)
                    row.cells[4].text = reason_text
                else:
                    row.cells[1].text = "No quotes found"
                    row.cells[2].text = "No citation available"
                    row.cells[3].text = "No sources available"
                    row.cells[4].text = "No reason available"
                
                # Set font size for all cells in this row
                for cell in row.cells:
                    for para in cell.paragraphs:
                        for run in para.runs:
                            run.font.size = Inches(0.08)  # Smaller font for better fit
            
            # Add summary paragraph
            total_quotes = sum(1 for concept in all_concepts 
                             if self._get_detailed_quotes_for_concept(concept, df, file_to_varying_param)['quotes'])
            processing_time = time.time() - start_time
            doc.add_paragraph(
                f"Summary: Found quotes for {total_quotes} out of {len(all_concepts)} concepts. Processing time: {processing_time:.2f} seconds"
            )
            
            print(f"[QUOTES STATS] Successfully added quotes section with {len(all_concepts)} concepts in {processing_time:.2f} seconds")
            
        except Exception as e:
            print(f"[QUOTES STATS ERROR] Failed to add quotes section: {e}")
            import traceback
            traceback.print_exc()

    def _extract_quotes_for_concept_from_csvs(self, concept_name, outdir):
        """Extract quotes for a specific concept from CSV files in the output directory"""
        quotes = []
        
        try:
            import pandas as pd
            import re
            import os
            
            # Look for CSV files in the output directory
            csv_files = [f for f in os.listdir(outdir) if f.endswith('.csv')]
            print(f"[QUOTES DEBUG] Looking for quotes for '{concept_name}' in {len(csv_files)} CSV files")
            
            for csv_file in csv_files:
                csv_path = os.path.join(outdir, csv_file)
                
                try:
                    df = pd.read_csv(csv_path)
                    print(f"[QUOTES DEBUG] Processing CSV {csv_file} with {len(df)} rows")
                    
                    # Look for Main Answer column
                    if 'Main Answer' in df.columns:
                        for _, row in df.iterrows():
                            content = str(row['Main Answer'])
                            if content and content != 'nan':
                                # Extract specific use sections for this concept
                                specific_uses = self._extract_specific_uses_from_content(content, concept_name)
                                if specific_uses:
                                    quotes.extend(specific_uses)
                                    print(f"[QUOTES DEBUG] Found {len(specific_uses)} quotes in {csv_file}")
                    else:
                        print(f"[QUOTES DEBUG] No 'Main Answer' column in {csv_file}")
                
                except Exception as e:
                    print(f"[QUOTES ERROR] Error reading CSV {csv_file}: {e}")
            
            # Remove duplicates and limit quotes
            quotes = list(dict.fromkeys(quotes))[:3]  # Keep first 3 unique quotes
            print(f"[QUOTES DEBUG] Final quotes for '{concept_name}': {len(quotes)}")
            
        except Exception as e:
            print(f"[QUOTES ERROR] Error extracting quotes for concept '{concept_name}': {e}")
            import traceback
            traceback.print_exc()
        
        return quotes

    def _extract_specific_uses_from_content(self, content, concept_name):
        """Extract specific use sections for a concept from content"""
        specific_uses = []
        
        try:
            import re
            
            # Try multiple patterns to find the concept and its quotes
            patterns_to_try = [
                # Exact match with **Concept**
                rf'\*\*{re.escape(concept_name)}\*\*.*?\*Specific Use\*:\s*"([^"]+)"',
                # Partial match (main concept without parentheses)
                rf'\*\*{re.escape(concept_name.split("(")[0].strip())}\*\*.*?\*Specific Use\*:\s*"([^"]+)"',
                # Any quoted text near the concept
                rf'\*\*{re.escape(concept_name)}\*\*.*?"([^"]+)"',
                # Partial match with any quoted text
                rf'\*\*{re.escape(concept_name.split("(")[0].strip())}\*\*.*?"([^"]+)"',
            ]
            
            for pattern in patterns_to_try:
                matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    quote = match.strip()
                    if len(quote) > 10:  # Filter out very short quotes
                        specific_uses.append(quote)
                
                if specific_uses:  # If we found quotes, stop trying other patterns
                    break
            
            # If still no quotes, try a more general approach
            if not specific_uses:
                # Look for any text that contains the concept name and has quotes
                concept_words = concept_name.split()
                if len(concept_words) > 0:
                    # Try with the first word of the concept
                    first_word = concept_words[0]
                    pattern = rf'{re.escape(first_word)}.*?"([^"]+)"'
                    matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
                    for match in matches:
                        quote = match.strip()
                        if len(quote) > 20:  # Longer minimum for general quotes
                            specific_uses.append(quote)
            
            print(f"[QUOTES DEBUG] Extracted {len(specific_uses)} specific uses for '{concept_name}'")
            
        except Exception as e:
            print(f"[QUOTES ERROR] Error extracting specific uses: {e}")
        
        return specific_uses

    def _get_quotes_for_concept_from_data(self, concept_name, df):
        """Get quotes for a concept from the stored Concept_Quotes data"""
        quotes = []
        
        try:
            print(f"[QUOTES DEBUG] Looking for quotes for '{concept_name}' in dataframe with {len(df)} rows")
            
            # Check if Concept_Quotes column exists
            if 'Concept_Quotes' not in df.columns:
                print(f"[QUOTES DEBUG] Concept_Quotes column not found in dataframe")
                return quotes
            
            # Look through all rows in the dataframe
            for i, row in df.iterrows():
                if 'Concept_Quotes' in row and isinstance(row['Concept_Quotes'], dict):
                    concept_quotes = row['Concept_Quotes']
                    print(f"[QUOTES DEBUG] Row {i} has {len(concept_quotes)} concept quotes: {list(concept_quotes.keys())}")
                    if concept_name in concept_quotes:
                        quotes.extend(concept_quotes[concept_name])
                        print(f"[QUOTES DEBUG] Found quotes for '{concept_name}' in row {i}: {concept_quotes[concept_name]}")
                else:
                    print(f"[QUOTES DEBUG] Row {i} has no Concept_Quotes or it's not a dict: {type(row.get('Concept_Quotes', 'Not found'))}")
            
            # Remove duplicates and limit quotes
            quotes = list(dict.fromkeys(quotes))[:3]  # Keep first 3 unique quotes
            print(f"[QUOTES DEBUG] Found {len(quotes)} quotes for '{concept_name}' in stored data")
            
        except Exception as e:
            print(f"[QUOTES ERROR] Error getting quotes for concept '{concept_name}': {e}")
            import traceback
            traceback.print_exc()
        
        return quotes

    def _get_detailed_quotes_for_concept(self, concept_name, df, file_to_varying_param=None):
        """Get detailed quote information including source parameters and context"""
        quote_info = {
            'quotes': [],
            'sources': [],
            'reason_text': '',
            'specific_use': ''
        }
        
        try:
            print(f"[QUOTES DEBUG] Getting detailed quotes for '{concept_name}'")
            
            # Check if Concept_Quotes column exists
            if 'Concept_Quotes' not in df.columns:
                print(f"[QUOTES DEBUG] Concept_Quotes column not found in dataframe")
                return quote_info
            
            # Look through all rows in the dataframe where the concept appears
            for i, row in df.iterrows():
                # Check if concept appears in the Concepts column
                if 'Concepts' in row and concept_name in row['Concepts']:
                    # Extract quotes if available
                    if 'Concept_Quotes' in row and isinstance(row['Concept_Quotes'], dict):
                        concept_quotes = row['Concept_Quotes']
                        if concept_name in concept_quotes:
                            quotes = concept_quotes[concept_name]
                            quote_info['quotes'].extend(quotes)
                            
                            # Extract source information with file tracking
                            source_info = {
                                'file': row.get('File', 'Unknown'),
                                'temperature': row.get('Temperature', 'N/A'),
                                'top_p': row.get('Top-p', 'N/A'),
                                'top_k': row.get('Top-k', 'N/A'),
                                'bm25_weight': row.get('BM25 Weight', 'N/A'),
                                'main_answer': row.get('Main Answer', ''),
                                'varying_param': None  # Will be set based on file
                            }
                            
                            # Determine varying parameter based on file
                            if file_to_varying_param and source_info['file'] in file_to_varying_param:
                                source_info['varying_param'] = file_to_varying_param[source_info['file']]
                                print(f"[QUOTES DEBUG] Source from file '{source_info['file']}' has varying param: {source_info['varying_param']}")
                            
                            quote_info['sources'].append(source_info)
                            
                            # Extract reason and specific use from Main Answer
                            reason_info = self._extract_reason_and_specific_use(row.get('Main Answer', ''), concept_name)
                            if reason_info['reason']:
                                quote_info['reason_text'] = reason_info['reason']
                            if reason_info['specific_use']:
                                quote_info['specific_use'] = reason_info['specific_use']
            
            # Remove duplicates and limit quotes
            quote_info['quotes'] = list(dict.fromkeys(quote_info['quotes']))[:3]
            print(f"[QUOTES DEBUG] Found {len(quote_info['quotes'])} quotes and {len(quote_info['sources'])} sources for '{concept_name}'")
            
        except Exception as e:
            print(f"[QUOTES ERROR] Error getting detailed quotes for concept '{concept_name}': {e}")
            import traceback
            traceback.print_exc()
        
        return quote_info

    def _extract_reason_and_specific_use(self, text, concept_name):
        """Extract reason for selection and specific use from text"""
        reason_info = {'reason': '', 'specific_use': ''}
        
        try:
            import re
            
            print(f"[QUOTES DEBUG] Extracting reason for concept '{concept_name}' from text of length {len(text)}")
            
            # Look for the concept and extract reason and specific use
            # Pattern: **Concept** ... **Reason for Selection**: ... **Specific Use**: ...
            pattern = rf'\*\*{re.escape(concept_name)}\*\*.*?\*\*Reason for Selection\*\*:\s*([^*]+?)(?:\*\*Specific Use\*\*:\s*([^*]+))?'
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            
            if match:
                reason_info['reason'] = match.group(1).strip()
                if match.group(2):
                    reason_info['specific_use'] = match.group(2).strip()
                print(f"[QUOTES DEBUG] Found reason: {reason_info['reason'][:100]}...")
            else:
                # Try alternative patterns - look for reason and specific use separately
                # Look for just reason - use a more permissive pattern
                reason_pattern = rf'\*\*{re.escape(concept_name)}\*\*.*?\*\*Reason for Selection\*\*:\s*([^*]+?)(?=\*\*|$)'
                reason_match = re.search(reason_pattern, text, re.DOTALL | re.IGNORECASE)
                if reason_match:
                    reason_text = reason_match.group(1).strip()
                    # Clean up the reason text
                    reason_text = re.sub(r'\s+', ' ', reason_text)  # Replace multiple spaces
                    reason_info['reason'] = reason_text
                    print(f"[QUOTES DEBUG] Found reason (alt): {reason_info['reason'][:100]}...")
                
                # Look for just specific use - use a more permissive pattern
                specific_pattern = rf'\*\*{re.escape(concept_name)}\*\*.*?\*\*Specific Use\*\*:\s*([^*]+?)(?=\*\*|$)'
                specific_match = re.search(specific_pattern, text, re.DOTALL | re.IGNORECASE)
                if specific_match:
                    specific_text = specific_match.group(1).strip()
                    # Clean up the specific use text
                    specific_text = re.sub(r'\s+', ' ', specific_text)  # Replace multiple spaces
                    reason_info['specific_use'] = specific_text
                    print(f"[QUOTES DEBUG] Found specific use: {reason_info['specific_use'][:100]}...")
            
            # If still no reason found, try a broader search
            if not reason_info['reason']:
                # Look for any text that might contain reason information
                broader_pattern = rf'{re.escape(concept_name)}.*?Reason for Selection.*?:\s*([^.]+\.[^.]*)'
                broader_match = re.search(broader_pattern, text, re.DOTALL | re.IGNORECASE)
                if broader_match:
                    reason_text = broader_match.group(1).strip()
                    reason_text = re.sub(r'\s+', ' ', reason_text)  # Clean up
                    reason_info['reason'] = reason_text
                    print(f"[QUOTES DEBUG] Found reason (broader): {reason_info['reason'][:100]}...")
            
        except Exception as e:
            print(f"[QUOTES ERROR] Error extracting reason and specific use: {e}")
            import traceback
            traceback.print_exc()
        
        return reason_info

    def _format_full_citation(self, quote_info):
        """Format full citation with all details"""
        if not quote_info['quotes'] or not quote_info['sources']:
            return "No citation available"
        
        citation_parts = []
        
        # Add all quotes (no truncation)
        for i, quote in enumerate(quote_info['quotes']):
            citation_parts.append(f"Quote {i+1}: {quote}")
        
        # Add source information
        for i, source in enumerate(quote_info['sources']):
            source_text = f"Source {i+1}: Temp={source['temperature']}, Top-p={source['top_p']}, Top-k={source['top_k']}, BM25={source['bm25_weight']}"
            citation_parts.append(source_text)
        
        return "\n\n".join(citation_parts)

    def _format_quotes_only(self, quote_info):
        """Format quotes without source information"""
        if not quote_info['quotes']:
            return "No quotes available"
        
        quote_parts = []
        
        # Add all quotes (no truncation)
        for i, quote in enumerate(quote_info['quotes']):
            quote_parts.append(f"Quote {i+1}: {quote}")
        
        return "\n\n".join(quote_parts)

    def _format_sources_info(self, quote_info):
        """Format source parameter information"""
        if not quote_info['sources']:
            return "No sources available"
        
        source_parts = []
        
        # Add source information
        for i, source in enumerate(quote_info['sources']):
            source_text = f"Source {i+1}: Temp={source['temperature']}, Top-p={source['top_p']}, Top-k={source['top_k']}, BM25={source['bm25_weight']}"
            source_parts.append(source_text)
        
        return "\n\n".join(source_parts)

    def _detect_varying_parameter_from_sources(self, sources):
        """Detect which parameter is actually varying in the sources data by finding consecutive sweeps"""
        if not sources or len(sources) < 2:
            return None
        
        # Extract parameter values
        param_values = {
            'Temperature': [],
            'Top-p': [],
            'Top-k': [],
            'BM25 Weight': []
        }
        
        for source in sources:
            param_values['Temperature'].append(source.get('temperature', 'N/A'))
            param_values['Top-p'].append(source.get('top_p', 'N/A'))
            param_values['Top-k'].append(source.get('top_k', 'N/A'))
            param_values['BM25 Weight'].append(source.get('bm25_weight', 'N/A'))
        
        # Convert to comparable values
        clean_param_values = {}
        for param_name, values in param_values.items():
            clean_values = []
            for val in values:
                if val == 'N/A' or val is None:
                    clean_values.append(None)
                else:
                    try:
                        clean_values.append(float(val))
                    except (ValueError, TypeError):
                        clean_values.append(str(val))
            clean_param_values[param_name] = clean_values
        
        print(f"[DEBUG] Analyzing {len(sources)} sources for parameter sweeps:")
        for param_name, values in clean_param_values.items():
            non_none = [v for v in values if v is not None]
            unique_vals = set(non_none)
            print(f"[DEBUG] {param_name}: {values} -> {len(unique_vals)} unique values: {sorted(unique_vals)}")
        
        # Find consecutive parameter sweeps
        sweep_detections = []
        
        for param_name, values in clean_param_values.items():
            # Find consecutive ranges where this parameter varies significantly
            sweep_ranges = self._find_consecutive_sweeps(values)
            print(f"[DEBUG] {param_name} sweep ranges: {sweep_ranges}")
            
            for start, end in sweep_ranges:
                # Check if other parameters are relatively constant in this range
                other_params_constant = True
                for other_param, other_values in clean_param_values.items():
                    if other_param == param_name:
                        continue
                    
                    # Check if other parameter values in this range are relatively constant
                    range_values = other_values[start:end]
                    non_none_range = [v for v in range_values if v is not None]
                    if len(non_none_range) > 1:
                        # Calculate coefficient of variation
                        try:
                            import statistics
                            mean_val = statistics.mean(non_none_range)
                            if mean_val != 0:
                                std_val = statistics.stdev(non_none_range)
                                cv = std_val / abs(mean_val)
                                # If coefficient of variation > 0.1, consider it varying
                                if cv > 0.1:
                                    other_params_constant = False
                                    print(f"[DEBUG] {other_param} is varying in range {start}-{end} (cv={cv:.3f})")
                                    break
                        except (statistics.StatisticsError, ZeroDivisionError):
                            other_params_constant = False
                            print(f"[DEBUG] {other_param} calculation error in range {start}-{end}")
                            break
                
                if other_params_constant and (end - start) >= 3:  # At least 3 consecutive sources
                    sweep_detections.append((param_name, end - start))
                    print(f"[DEBUG] Found sweep for {param_name} in range {start}-{end}")
        
        # Return the parameter that appears in the most sweep detections
        if sweep_detections:
            param_counts = {}
            for param, count in sweep_detections:
                param_counts[param] = param_counts.get(param, 0) + count
            print(f"[DEBUG] Sweep detections: {sweep_detections}")
            print(f"[DEBUG] Param counts: {param_counts}")
            result = max(param_counts, key=param_counts.get)
            print(f"[DEBUG] Selected parameter: {result}")
            return result
        
        # Fallback to simple variation detection
        param_scores = {}
        for param_name, values in clean_param_values.items():
            non_none_values = [v for v in values if v is not None]
            if len(non_none_values) > 1 and len(set(non_none_values)) > 1:
                variation_score = len(set(non_none_values)) / len(non_none_values)
                param_scores[param_name] = variation_score
        
        print(f"[DEBUG] Fallback varying_params: {param_scores}")
        if param_scores:
            result = max(param_scores, key=lambda x: x[1])[0]
            print(f"[DEBUG] Fallback selected parameter: {result}")
            return result
        
        print(f"[DEBUG] No varying parameter detected")
        return None

    def _find_consecutive_sweeps(self, values):
        """Find consecutive ranges where a parameter varies significantly"""
        ranges = []
        n = len(values)
        
        i = 0
        while i < n - 2:  # Need at least 3 values for a sweep
            if values[i] is None:
                i += 1
                continue
            
            # Find the end of a potential sweep group
            j = i + 1
            consecutive_varying = 0
            
            while j < n and values[j] is not None:
                # Check if this value is significantly different from the previous
                try:
                    if abs(float(values[j]) - float(values[j-1])) > 0.01:  # Significant change
                        consecutive_varying += 1
                        j += 1
                    else:
                        # If we have at least 2 consecutive varying values, this is a sweep
                        if consecutive_varying >= 2:  # At least 3 total values (including the first)
                            ranges.append((i, j))
                        break
                except (ValueError, TypeError):
                    break
            
            # Check if we reached the end with a valid sweep
            if j == n and consecutive_varying >= 2:
                ranges.append((i, j))
            
            i = j if j > i else i + 1
        
        return ranges

    def _find_parameter_sweep_ranges(self, values):
        """Find consecutive ranges where a parameter varies significantly"""
        ranges = []
        n = len(values)
        
        i = 0
        while i < n - 1:
            if values[i] is None:
                i += 1
                continue
            
            # Find the end of a potential sweep group
            j = i + 1
            consecutive_varying = 0
            
            while j < n and values[j] is not None:
                # Check if this value is significantly different from the previous
                try:
                    if abs(float(values[j]) - float(values[j-1])) > 0.01:  # Significant change
                        consecutive_varying += 1
                        j += 1
                    else:
                        # If we have at least 2 consecutive varying values, this is a sweep
                        if consecutive_varying >= 1:  # At least 2 total values (including the first)
                            ranges.append((i, j))
                        break
                except (ValueError, TypeError):
                    break
            
            # Check if we reached the end with a valid sweep
            if j == n and consecutive_varying >= 1:
                ranges.append((i, j))
            
            i = j if j > i else i + 1
        
        return ranges

    def _format_sources_cell_with_per_source_varying_param(self, cell, quote_info):
        """Format sources cell with per-source varying parameter detection and bolding"""
        if not quote_info['sources']:
            cell.text = "No sources available"
            return

        # Clear the cell first
        cell.text = ""

        # Parameter name mapping for display
        param_display_names = {
            'Temperature': 'TEMP',
            'Top-p': 'TOP P',
            'Top-k': 'TOP K',
            'BM25 Weight': 'BM25'
        }

        # Add source information with per-source varying parameter detection
        for i, source in enumerate(quote_info['sources']):
            if i > 0:
                # Add line break between sources
                cell.paragraphs[0].add_run("\n\n")

            # Start with "Source X: "
            source_run = cell.paragraphs[0].add_run(f"Source {i+1}: ")

            # Get the varying parameter for this specific source
            source_varying_param = source.get('varying_param', 'Unknown')
            print(f"[DEBUG] Source {i+1} varying param: {source_varying_param}")

            # Add parameter values with bold for the varying parameter
            params = [
                ('Temp', source['temperature'], 'Temperature'),
                ('Top-p', source['top_p'], 'Top-p'),
                ('Top-k', source['top_k'], 'Top-k'),
                ('BM25', source['bm25_weight'], 'BM25 Weight')
            ]

            for j, (param_name, param_value, param_key) in enumerate(params):
                if j > 0:
                    cell.paragraphs[0].add_run(", ")

                # Add parameter name and equals sign
                cell.paragraphs[0].add_run(f"{param_name}=")

                # Add parameter value - bold if it's the varying parameter for this source
                if param_key == source_varying_param:
                    bold_run = cell.paragraphs[0].add_run(str(param_value))
                    bold_run.bold = True
                    print(f"[DEBUG] Bolded {param_name}={param_value} for source {i+1}")
                else:
                    cell.paragraphs[0].add_run(str(param_value))

            # Add parameter name in parentheses
            param_display = param_display_names.get(source_varying_param, source_varying_param)
            cell.paragraphs[0].add_run(f" ({param_display} Sweep)")

    def _format_sources_cell_with_bold(self, cell, quote_info, varying_param):
        """Format sources cell with bold varying parameter value and parameter name in parentheses"""
        if not quote_info['sources']:
            cell.text = "No sources available"
            return

        # Clear the cell first
        cell.text = ""

        # Parameter name mapping for display
        param_display_names = {
            'Temperature': 'TEMP',
            'Top-p': 'TOP P',
            'Top-k': 'TOP K',
            'BM25 Weight': 'BM25'
        }

        # Detect the actual varying parameter from the data
        actual_varying_param = self._detect_varying_parameter_from_sources(quote_info['sources'])
        print(f"[DEBUG] File-based varying_param: {varying_param}")
        print(f"[DEBUG] Data-based actual_varying_param: {actual_varying_param}")

        # Use the actual varying parameter if detected, otherwise fall back to file-based detection
        final_varying_param = actual_varying_param if actual_varying_param else varying_param
        param_display = param_display_names.get(final_varying_param, final_varying_param)
        print(f"[DEBUG] Final varying_param: {final_varying_param}, display: {param_display}")

        # Add source information with formatting
        for i, source in enumerate(quote_info['sources']):
            if i > 0:
                # Add line break between sources
                cell.paragraphs[0].add_run("\n\n")

            # Start with "Source X: "
            source_run = cell.paragraphs[0].add_run(f"Source {i+1}: ")

            # Add parameter values with bold for the varying parameter
            params = [
                ('Temp', source['temperature'], 'Temperature'),
                ('Top-p', source['top_p'], 'Top-p'),
                ('Top-k', source['top_k'], 'Top-k'),
                ('BM25', source['bm25_weight'], 'BM25 Weight')
            ]

            for j, (param_name, param_value, param_key) in enumerate(params):
                if j > 0:
                    cell.paragraphs[0].add_run(", ")

                # Add parameter name and equals sign
                cell.paragraphs[0].add_run(f"{param_name}=")

                # Add parameter value - bold if it's the varying parameter
                if param_key == final_varying_param:
                    bold_run = cell.paragraphs[0].add_run(str(param_value))
                    bold_run.bold = True
                else:
                    cell.paragraphs[0].add_run(str(param_value))

            # Add parameter name in parentheses
            cell.paragraphs[0].add_run(f" ({param_display} Sweep)")

    def _format_reason_text(self, quote_info):
        """Format reason text with reason for selection and specific use for each quote"""
        reason_parts = []
        
        # Ensure quote_info has all required keys (backward compatibility)
        if 'reason_text' not in quote_info:
            quote_info['reason_text'] = ''
        if 'specific_use' not in quote_info:
            quote_info['specific_use'] = ''
        
        # Add general reason for selection if available and not truncated
        if quote_info.get('reason_text') and len(quote_info['reason_text'].strip()) > 5:
            reason_parts.append(f"**Reason for Selection**: {quote_info['reason_text']}")
        
        # Safely check for specific_use key (may not exist in older data)
        if quote_info.get('specific_use') and len(quote_info['specific_use'].strip()) > 5:
            reason_parts.append(f"**Specific Use**: {quote_info['specific_use']}")
        
        # Add individual reasons for each quote
        if quote_info['quotes'] and quote_info['sources']:
            for i, (quote, source) in enumerate(zip(quote_info['quotes'], quote_info['sources'])):
                # Try to extract reason for this specific quote from the source text
                quote_reason = self._extract_reason_for_quote(quote, source['main_answer'])
                if quote_reason and len(quote_reason.strip()) > 10:  # Only add if meaningful
                    reason_parts.append(f"**Quote {i+1} Reason**: {quote_reason}")
        
        if not reason_parts:
            return "No reason information available"
        
        return "\n\n".join(reason_parts)

    def _extract_reason_for_quote(self, quote, text):
        """Extract reason for a specific quote from the text"""
        try:
            import re
            
            print(f"[QUOTES DEBUG] Extracting reason for quote: {quote[:50]}...")
            
            # Look for the quote in the text and extract surrounding context
            # Escape special regex characters in the quote
            escaped_quote = re.escape(quote)
            
            # Try multiple patterns to get better context extraction
            patterns_to_try = [
                # Pattern 1: Extract more context around the quote (multiple sentences)
                rf'([^.]*?{escaped_quote}[^.]*?[^.]*?[^.]*)',
                # Pattern 2: Extract text before and after the quote (up to 2 sentences each side)
                rf'([^.]*?[^.]*?{escaped_quote}[^.]*?[^.]*?)',
                # Pattern 3: Look for the quote and extract a larger context window
                rf'([^.]*?{escaped_quote}[^.]*)',
                # Pattern 4: Simple pattern with partial quote match
                rf'([^.]*{re.escape(quote[:30])}[^.]*)',
            ]
            
            for i, pattern in enumerate(patterns_to_try):
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    context = match.group(1).strip()
                    # Clean up the context
                    context = re.sub(r'\s+', ' ', context)  # Replace multiple spaces with single space
                    
                    # Remove any markdown formatting that might interfere
                    context = re.sub(r'\*\*([^*]+)\*\*', r'\1', context)  # Remove **bold**
                    context = re.sub(r'\*([^*]+)\*', r'\1', context)  # Remove *italic*
                    
                    # Only return if we have substantial context
                    if len(context) > 50:  # Minimum meaningful context
                        print(f"[QUOTES DEBUG] Found context (pattern {i+1}): {context[:100]}...")
                        return context
            
            # If no match found, try a broader approach
            # Look for any text that contains part of the quote
            if len(quote) > 20:
                partial_quote = quote[:20]
                broader_pattern = rf'([^.]*{re.escape(partial_quote)}[^.]*)'
                broader_match = re.search(broader_pattern, text, re.DOTALL | re.IGNORECASE)
                
                if broader_match:
                    context = broader_match.group(1).strip()
                    context = re.sub(r'\s+', ' ', context)
                    context = re.sub(r'\*\*([^*]+)\*\*', r'\1', context)
                    context = re.sub(r'\*([^*]+)\*', r'\1', context)
                    if len(context) > 50:
                        print(f"[QUOTES DEBUG] Found context (broader): {context[:100]}...")
                        return context
            
        except Exception as e:
            print(f"[QUOTES ERROR] Error extracting reason for quote: {e}")
            import traceback
            traceback.print_exc()
        
        return None

    def _extract_simple_quotes_for_concept(self, concept_name, outdir):
        """Simple fallback method to extract any text containing the concept"""
        try:
            import pandas as pd
            import os
            
            # Look for CSV files in the output directory
            csv_files = [f for f in os.listdir(outdir) if f.endswith('.csv')]
            
            for csv_file in csv_files:
                csv_path = os.path.join(outdir, csv_file)
                
                try:
                    df = pd.read_csv(csv_path)
                    
                    # Look for any column that might contain the concept
                    for col in df.columns:
                        if col in ['Main Answer', 'Specific Use', 'Text', 'Content']:
                            for _, row in df.iterrows():
                                content = str(row[col])
                                if content and content != 'nan' and concept_name.lower() in content.lower():
                                    # Extract a reasonable snippet around the concept
                                    words = content.split()
                                    concept_words = concept_name.split()
                                    if len(concept_words) > 0:
                                        first_word = concept_words[0].lower()
                                        for i, word in enumerate(words):
                                            if first_word in word.lower():
                                                # Extract 20 words before and after
                                                start = max(0, i - 20)
                                                end = min(len(words), i + 20)
                                                snippet = ' '.join(words[start:end])
                                                if len(snippet) > 50:  # Only return substantial snippets
                                                    return snippet
                
                except Exception as e:
                    print(f"[QUOTES ERROR] Error reading CSV {csv_file}: {e}")
            
        except Exception as e:
            print(f"[QUOTES ERROR] Error in simple quotes extraction: {e}")
        
        return None

    def _read_quotes_from_stats_docs(self, valid_folders):
        """Read quotes from compare_stats.docx files in each folder"""
        quotes_data = {}
        
        try:
            import docx
            import os
            
            # Get all unique concepts across all folders
            all_concepts = set()
            
            # First pass: collect all concepts from all stats docs
            for folder in valid_folders:
                folder_name = os.path.basename(folder)
                stats_doc_path = os.path.join(folder, "compare_stats.docx")
                
                if os.path.exists(stats_doc_path):
                    print(f"[QUOTES] Processing {folder_name}/compare_stats.docx")
                    concepts = self._extract_concepts_from_stats_doc(stats_doc_path)
                    all_concepts.update(concepts)
                    print(f"[QUOTES] Found {len(concepts)} concepts in {folder_name}/compare_stats.docx")
                    if len(concepts) <= 5:
                        print(f"[QUOTES] Concepts for {folder_name}: {concepts}")
                else:
                    print(f"[QUOTES] No compare_stats.docx found in {folder_name}")
            
            all_concepts = sorted(list(all_concepts))
            print(f"[QUOTES] Processing {len(all_concepts)} concepts across {len(valid_folders)} folders")
            
            # Second pass: extract quotes and sources for each concept from each folder
            for concept in all_concepts:
                quotes_data[concept] = {}
                
                for folder in valid_folders:
                    folder_name = os.path.basename(folder)
                    quotes_data[concept][folder_name] = {
                        'quotes': [],
                        'sources': []
                    }
                    
                    stats_doc_path = os.path.join(folder, "compare_stats.docx")
                    if os.path.exists(stats_doc_path):
                        print(f"[QUOTES DEBUG] Extracting quotes for '{concept}' from {folder_name}")
                        quotes = self._extract_quotes_for_concept_from_stats_doc(stats_doc_path, concept)
                        sources = self._extract_sources_for_concept_from_stats_doc(stats_doc_path, concept)
                        quotes_data[concept][folder_name]['quotes'] = quotes
                        quotes_data[concept][folder_name]['sources'] = sources
                        if quotes:
                            print(f"[QUOTES] Found {len(quotes)} quotes and {len(sources)} sources for '{concept}' in {folder_name}")
                            print(f"[QUOTES DEBUG] Quotes: {quotes[:2]}")  # Show first 2 quotes
                        elif len(quotes_data[concept]) == 1:  # Only print for first folder to avoid spam
                            print(f"[QUOTES] No quotes found for '{concept}' in any folder")
            
        except Exception as e:
            print(f"[QUOTES ERROR] Failed to read quotes from stats docs: {e}")
            import traceback
            traceback.print_exc()
        
        return quotes_data

    def _extract_concepts_from_stats_doc(self, stats_doc_path):
        """Extract concept names from a compare_stats.docx file"""
        concepts = []
        
        try:
            import docx
            
            doc = docx.Document(stats_doc_path)
            
            # Look for the "Concept Citations and Quotes" section
            in_quotes_section = False
            
            for paragraph in doc.paragraphs:
                # Check if we're in the quotes section
                if paragraph.text.strip() == "Concept Citations and Quotes":
                    in_quotes_section = True
                    continue
                elif in_quotes_section and paragraph.text.strip().startswith("Summary:"):
                    # End of quotes section
                    break
                elif in_quotes_section and paragraph.text.strip():
                    # Skip header paragraphs
                    if paragraph.text.strip() in ["Concept", "Short Quote", "Full Citation", "Sources", "Reason"]:
                        continue
                    
                    # Skip descriptive paragraphs (long text that explains the table)
                    concept = paragraph.text.strip()
                    if (concept and 
                        not concept.startswith("No quotes found") and 
                        not concept.startswith("This table shows") and
                        not concept.startswith("This table") and
                        len(concept) < 200 and  # Skip very long paragraphs
                        self._is_valid_concept(concept)):
                        concepts.append(concept)
            
            # Check tables for concepts (new 5-column format)
            for table in doc.tables:
                for row in table.rows:
                    if len(row.cells) >= 5:  # New format has 5 columns
                        concept_cell = row.cells[0]  # First column is Concept
                        concept_text = concept_cell.text.strip()
                        
                        if concept_text and concept_text not in ["Concept", "Short Quote", "Full Citation", "Sources", "Reason"] and self._is_valid_concept(concept_text):
                            concepts.append(concept_text)
                    elif len(row.cells) >= 2:  # Fallback for old format
                        concept_cell = row.cells[0]
                        quotes_cell = row.cells[1]
                        
                        concept_text = concept_cell.text.strip()
                        quotes_text = quotes_cell.text.strip()
                        
                        if concept_text and concept_text not in ["Concept", "Quotes and Citations"] and self._is_valid_concept(concept_text):
                            concepts.append(concept_text)
            
            print(f"[QUOTES DEBUG] Extracted {len(concepts)} concepts from {os.path.basename(stats_doc_path)}")
            if concepts:
                print(f"[QUOTES DEBUG] First 3 concepts: {concepts[:3]}")
            else:
                print(f"[QUOTES DEBUG] No concepts found in {os.path.basename(stats_doc_path)}")
                # Debug: print all paragraph texts to see what's in the document
                print(f"[QUOTES DEBUG] Document paragraphs:")
                for i, para in enumerate(doc.paragraphs[:10]):  # First 10 paragraphs
                    print(f"  Para {i}: '{para.text.strip()}'")
                print(f"[QUOTES DEBUG] Document tables: {len(doc.tables)}")
                for i, table in enumerate(doc.tables):
                    print(f"  Table {i}: {len(table.rows)} rows, {len(table.columns)} columns")
                    if table.rows:
                        first_row = [cell.text.strip() for cell in table.rows[0].cells]
                        print(f"    First row: {first_row}")
            
        except Exception as e:
            print(f"[QUOTES ERROR] Error extracting concepts from {stats_doc_path}: {e}")
            import traceback
            traceback.print_exc()
        
        return list(set(concepts))  # Remove duplicates

    def _extract_quotes_for_concept_from_stats_doc(self, stats_doc_path, concept_name):
        """Extract quotes for a specific concept from a compare_stats.docx file"""
        quotes = []
        
        try:
            import docx
            
            doc = docx.Document(stats_doc_path)
            print(f"[QUOTES DEBUG] Looking for concept '{concept_name}' in {len(doc.tables)} tables")
            
            # Look for the concept in tables
            for table_idx, table in enumerate(doc.tables):
                for row in table.rows:
                    if len(row.cells) >= 5:  # New 5-column format
                        concept_cell = row.cells[0]  # Concept column
                        short_quote_cell = row.cells[1]  # Short Quote column
                        full_citation_cell = row.cells[2]  # Full Citation column
                        
                        concept_text = concept_cell.text.strip()
                        short_quote_text = short_quote_cell.text.strip()
                        full_citation_text = full_citation_cell.text.strip()
                        
                        if concept_text == concept_name:
                            print(f"[QUOTES DEBUG] Found matching concept '{concept_name}' in table {table_idx}")
                            print(f"[QUOTES DEBUG] Short quote: '{short_quote_text}'")
                            print(f"[QUOTES DEBUG] Full citation: '{full_citation_text}'")
                            # Use full citation if available, otherwise short quote
                            if full_citation_text and full_citation_text != "No citation available":
                                # Split quotes by semicolons (as formatted in the new system)
                                quote_list = [q.strip() for q in full_citation_text.split(';') if q.strip()]
                                quotes.extend(quote_list)
                                print(f"[QUOTES DEBUG] Added {len(quote_list)} quotes from full citation")
                            elif short_quote_text and short_quote_text != "No quotes found":
                                quotes.append(short_quote_text)
                                print(f"[QUOTES DEBUG] Added 1 quote from short quote")
                    elif len(row.cells) >= 2:  # Fallback for old format
                        concept_cell = row.cells[0]
                        quotes_cell = row.cells[1]
                        
                        concept_text = concept_cell.text.strip()
                        quotes_text = quotes_cell.text.strip()
                        
                        if concept_text == concept_name:
                            print(f"[QUOTES DEBUG] Found matching concept '{concept_name}' in old format table {table_idx}")
                            print(f"[QUOTES DEBUG] Quotes text: '{quotes_text}'")
                            if quotes_text and quotes_text != "No quotes found":
                                # Split quotes by double line breaks
                                quote_list = [q.strip() for q in quotes_text.split('\n\n') if q.strip()]
                                quotes.extend(quote_list)
                                print(f"[QUOTES DEBUG] Added {len(quote_list)} quotes from old format")
            
        except Exception as e:
            print(f"[QUOTES ERROR] Error extracting quotes for '{concept_name}' from {stats_doc_path}: {e}")
        
        return quotes

    def _extract_sources_for_concept_from_stats_doc(self, stats_doc_path, concept_name):
        """Extract sources information for a specific concept from a compare_stats.docx file"""
        sources = []
        
        try:
            import docx
            
            doc = docx.Document(stats_doc_path)
            
            # Look for the concept in tables
            for table in doc.tables:
                for row in table.rows:
                    if len(row.cells) >= 5:  # New 5-column format
                        concept_cell = row.cells[0]  # Concept column
                        sources_cell = row.cells[3]  # Sources column
                        
                        concept_text = concept_cell.text.strip()
                        sources_text = sources_cell.text.strip()
                        
                        if concept_text == concept_name and sources_text and sources_text != "No sources available":
                            # Split sources by double line breaks
                            source_list = [s.strip() for s in sources_text.split('\n\n') if s.strip()]
                            sources.extend(source_list)
                    elif len(row.cells) >= 2:  # Fallback for old format
                        # Old format doesn't have sources column
                        pass
            
        except Exception as e:
            print(f"[QUOTES ERROR] Error extracting sources for '{concept_name}' from {stats_doc_path}: {e}")
        
        return sources

    def on_group_by_subletters(self):
        if self.group_by_subletters.get():
            self.group_by_words.set(False)

    def on_group_by_words(self):
        if self.group_by_words.get():
            self.group_by_subletters.set(False)

    def on_color_criteria_toggle(self):
        if self.use_colors.get():
            self.group_by_same_color_cb.state(['!disabled'])
        else:
            self.group_by_same_color_cb.state(['disabled'])
            self.group_by_same_color.set(False)

    def on_agg_group_by_subletters(self):
        if self.agg_group_by_subletters.get():
            self.agg_group_by_words.set(False)

    def on_agg_group_by_words(self):
        if self.agg_group_by_words.get():
            self.agg_group_by_subletters.set(False)

    def on_agg_color_criteria_toggle(self):
        # This method can be used for future functionality if needed
        pass

    def on_agg_enable_fuzzy_toggle(self):
        # Enable/disable fuzzy logic controls based on checkbox
        if self.agg_enable_fuzzy.get():
            self.sim_threshold_entry.config(state='normal')
            self.grouping_logic_combo.config(state='readonly')
        else:
            self.sim_threshold_entry.config(state='disabled')
            self.grouping_logic_combo.config(state='disabled')

    def get_aggregation_params(self):
        """Get aggregation parameters from GUI controls"""
        try:
            threshold = float(self.sim_threshold_var.get())
        except Exception:
            threshold = 0.85
        
        grouping_logic = self.grouping_logic_var.get()
        if not self.agg_enable_fuzzy.get():
            grouping_logic = "None"
        
        try:
            min_letters = int(self.agg_min_letters_var.get())
        except Exception:
            min_letters = 5
        
        return {
            'threshold': threshold,
            'grouping_logic': grouping_logic,
            'min_letters': min_letters,
            'use_colors': self.agg_use_colors.get(),
            'group_by_words': self.agg_group_by_words.get(),
            'group_by_subletters': self.agg_group_by_subletters.get()
        }

    def select_aggregate_folder(self):
        folder_path = filedialog.askdirectory(title="Select Parent Folder Containing Results")
        if folder_path:
            self.aggregate_folder = folder_path
            self.aggregate_folder_label.config(text=folder_path, foreground="black")
            self.aggregate_btn.config(state='normal')
            self.aggregate_status_label.config(text="Ready to aggregate.", foreground="blue")
        else:
            self.aggregate_folder = None
            self.aggregate_folder_label.config(text="No folder selected", foreground="gray")
            self.aggregate_btn.config(state='disabled')
            self.aggregate_status_label.config(text="", foreground="blue")

    def aggregate_results(self):
        # Run aggregation in a background thread to avoid blocking the UI
        thread = threading.Thread(target=self._aggregate_results_thread)
        thread.daemon = True
        thread.start()

    def _add_parameter_comparison_table(self, parent_cell, valid_folders):
        """Add a comprehensive parameter comparison table based on compare.htm data"""
        try:
            print("[PARAMETER COMPARISON] Function called successfully")
            print(f"[PARAMETER COMPARISON] Parent cell type: {type(parent_cell)}")
            print(f"[PARAMETER COMPARISON] Valid folders: {valid_folders}")
            print("[PARAMETER COMPARISON] Starting real implementation")
            
            # Implement real logic for first folder only
            self._create_real_parameter_table(parent_cell, valid_folders)
            
            # Try to read the compare.htm file
            compare_html_path = os.path.join(os.path.dirname(__file__), "CrispClean", "compare.htm")
            if not os.path.exists(compare_html_path):
                print(f"[PARAMETER COMPARISON] Compare HTML not found at {compare_html_path}")
                self._create_fallback_parameter_table(parent_cell, valid_folders)
                return
            
            from bs4 import BeautifulSoup
            with open(compare_html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            table = soup.find('table')
            if not table:
                print("[PARAMETER COMPARISON] No table found in HTML")
                self._create_fallback_parameter_table(parent_cell, valid_folders)
                return
            
            # Color to folder mapping (based on the color codes in the data)
            color_to_folder = {
                '#9932CC': 'Chase',
                '#556B2F': 'Kant', 
                '#6A5ACD': 'Mill',
                '#C71585': 'Nietzsche',
                '#800000': 'Rousseau',
                '#FFD700': 'Smith',
                '#006400': 'Ross',
                '#D2691E': 'Irwin',
                '#B22222': 'Crisp',
                '#2E8B57': 'Aristotle',
                '#A0522D': 'Plato',
                '#FF8C00': 'Hume',
                '#20B2AA': 'Bentham',
                '#8B4513': 'Rawls',
                '#FF4500': 'Kant_2',
                '#228B22': 'Mill_2',
                '#8B008B': 'Rousseau_2'
            }
            
            # Parse HTML table rows
            parameter_stats = {}
            rows = table.find_all('tr')
            
            print(f"[PARAMETER COMPARISON] Processing {len(rows)-1} data rows from HTML")
            
            for idx, row in enumerate(rows[1:], 1):  # Skip header row
                cells = row.find_all('td')
                if len(cells) < 5:
                    continue
                    
                temp = float(cells[0].text.strip())
                topp = float(cells[1].text.strip())
                topk = int(cells[2].text.strip())
                bm25 = float(cells[3].text.strip())
                
                # Parse concepts from the last cell
                concepts_cell = cells[4]
                concepts = []
                concept_boxes = concepts_cell.find_all('span', class_='concept-box')
                
                for box in concept_boxes:
                    concept_name = box.text.strip()
                    color = box.get('style', '').split('background-color: ')[1].split(';')[0] if 'background-color:' in box.get('style', '') else '#000000'
                    concepts.append((concept_name, color))
                
                print(f"[PARAMETER COMPARISON] Row {idx}: T={temp}, P={topp}, K={topk}, B={bm25}, Concepts={len(concepts)}")
                
                # Group concepts by folder
                folder_concepts = {}
                for concept, color in concepts:
                    folder = color_to_folder.get(color, 'Unknown')
                    if folder not in folder_concepts:
                        folder_concepts[folder] = set()
                    folder_concepts[folder].add(concept)
                
                print(f"[PARAMETER COMPARISON] Folders found: {list(folder_concepts.keys())}")
                
                # Calculate metrics for this parameter combination
                param_key = f"T{temp}_P{topp}_K{topk}_B{bm25}"
                parameter_stats[param_key] = {
                    'params': {'temp': temp, 'topp': topp, 'topk': topk, 'bm25': bm25},
                    'folder_concepts': folder_concepts,
                    'total_concepts': len(concepts)
                }
            
            print(f"[PARAMETER COMPARISON] Created {len(parameter_stats)} parameter combinations")
            
            # Create the full summary table
            self._create_full_parameter_table(parent_cell, parameter_stats, valid_folders)
            
        except Exception as e:
            print(f"[PARAMETER COMPARISON ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    def _create_full_parameter_table(self, parent_cell, parameter_stats, valid_folders):
        """Create the complete parameter comparison table with all metrics"""
        try:
            print(f"[FULL PARAMETER TABLE] Creating table with {len(parameter_stats)} combinations")
            print(f"[FULL PARAMETER TABLE] Valid folders: {valid_folders}")
            
            # Calculate number of columns: 1 (parameter) + 3 metrics per folder + 2 overall metrics
            num_cols = 1 + (len(valid_folders) * 3) + 2
            summary_table = parent_cell.add_table(rows=1, cols=num_cols)
            summary_table.style = 'Table Grid'
            
            # Set column widths
            for row in summary_table.rows:
                for i, cell in enumerate(row.cells):
                    if i == 0:  # Parameter name column
                        cell.width = Inches(1.2)
                    else:
                        cell.width = Inches(0.6)
            
            # Header row
            hdr_cells = summary_table.rows[0].cells
            hdr_cells[0].text = "Parameter"
            
            col_idx = 1
            # Add folder-specific headers
            for folder in valid_folders:
                hdr_cells[col_idx].text = f"{folder}\nF Total"
                col_idx += 1
                hdr_cells[col_idx].text = f"{folder}\nF Unique"
                col_idx += 1
                hdr_cells[col_idx].text = f"{folder}\nF Shared"
                col_idx += 1
            
            # Overall metrics columns
            hdr_cells[col_idx].text = "Overall\nA Unique"
            col_idx += 1
            hdr_cells[col_idx].text = "Overall\nA Shared"
            
            # Calculate and fill data for each parameter
            parameter_rows = [
                ('Temperature', 'temp'),
                ('Topp', 'topp'), 
                ('Topk', 'topk'),
                ('BM25', 'bm25')
            ]
            
            for param_name, param_key in parameter_rows:
                row_cells = summary_table.add_row().cells
                row_cells[0].text = param_name
                
                # Calculate metrics for this parameter
                self._calculate_full_parameter_metrics(row_cells, param_key, parameter_stats, valid_folders)
                
        except Exception as e:
            print(f"[FULL PARAMETER TABLE ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    def _calculate_full_parameter_metrics(self, row_cells, param_key, parameter_stats, valid_folders):
        """Calculate full metrics for a specific parameter"""
        try:
            col_idx = 1
            
            # Get all parameter combinations for this parameter type
            relevant_combinations = []
            for param_key_full, stats in parameter_stats.items():
                if param_key in stats['params']:
                    relevant_combinations.append(stats)
            
            print(f"[FULL METRICS] Found {len(relevant_combinations)} combinations for {param_key}")
            
            # Calculate overall unique and shared concepts across all combinations
            all_concepts = set()
            for combo in relevant_combinations:
                for folder_concepts in combo['folder_concepts'].values():
                    all_concepts.update(folder_concepts)
            
            print(f"[FULL METRICS] Total concepts across all combinations: {len(all_concepts)}")
            
            # For each folder, calculate metrics
            for folder in valid_folders:
                folder_concepts = set()
                for combo in relevant_combinations:
                    if folder in combo['folder_concepts']:
                        folder_concepts.update(combo['folder_concepts'][folder])
                
                # F Total - Number of concepts for this parameter in this folder
                f_total = len(folder_concepts)
                
                # F Unique - Concepts unique to this parameter in this folder
                f_unique = 0
                for concept in folder_concepts:
                    concept_count = sum(1 for combo in relevant_combinations 
                                      if folder in combo['folder_concepts'] and concept in combo['folder_concepts'][folder])
                    if concept_count == 1:
                        f_unique += 1
                
                # F Shared - Concepts shared by this parameter in this folder
                f_shared = f_total - f_unique
                
                # Fill the cells
                row_cells[col_idx].text = str(f_total)
                col_idx += 1
                row_cells[col_idx].text = str(f_unique)
                col_idx += 1
                row_cells[col_idx].text = str(f_shared)
                col_idx += 1
                
                print(f"[FULL METRICS] {folder}: Total={f_total}, Unique={f_unique}, Shared={f_shared}")
            
            # Calculate overall A metrics (across all folders)
            a_unique = 0
            a_shared = 0
            for concept in all_concepts:
                concept_count = sum(1 for combo in relevant_combinations 
                                  for folder_concepts in combo['folder_concepts'].values() 
                                  if concept in folder_concepts)
                if concept_count == 1:
                    a_unique += 1
                else:
                    a_shared += 1
            
            # Fill overall metrics
            row_cells[col_idx].text = str(a_unique)
            col_idx += 1
            row_cells[col_idx].text = str(a_shared)
            
            print(f"[FULL METRICS] Overall: Unique={a_unique}, Shared={a_shared}")
            
        except Exception as e:
            print(f"[FULL METRICS ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    def _create_real_parameter_table(self, parent_cell, valid_folders):
        """Create real parameter table with actual data for first folder only"""
        try:
            print("[REAL PARAMETER TABLE] Starting real implementation for first folder")
            
            # Get first folder (extract just the folder name from the path)
            first_folder_full = valid_folders[0] if valid_folders else "Unknown"
            first_folder = os.path.basename(first_folder_full) if first_folder_full != "Unknown" else "Unknown"
            print(f"[REAL PARAMETER TABLE] Using first folder: {first_folder} (from {first_folder_full})")
            
            # Try to read the compare.htm file
            compare_html_path = os.path.join(os.path.dirname(__file__), "CrispClean", "compare.htm")
            if not os.path.exists(compare_html_path):
                print(f"[REAL PARAMETER TABLE] Compare HTML not found at {compare_html_path}")
                return
            
            from bs4 import BeautifulSoup
            with open(compare_html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            table = soup.find('table')
            if not table:
                print("[REAL PARAMETER TABLE] No table found in HTML")
                return
            
            # Color to folder mapping (based on the color codes in the data)
            color_to_folder = {
                '#9932CC': 'Chase',
                '#556B2F': 'Kant', 
                '#6A5ACD': 'Mill',
                '#C71585': 'Nietzsche',
                '#800000': 'Rousseau',
                '#FFD700': 'Smith',
                '#006400': 'Ross',
                '#D2691E': 'Irwin',
                '#B22222': 'Crisp',
                '#2E8B57': 'Aristotle',
                '#A0522D': 'Plato',
                '#FF8C00': 'Hume',
                '#20B2AA': 'Bentham',
                '#8B4513': 'Rawls',
                '#FF4500': 'Kant_2',
                '#228B22': 'Mill_2',
                '#8B008B': 'Rousseau_2'
            }
            
            # Parse HTML table rows
            parameter_stats = {}
            rows = table.find_all('tr')
            
            print(f"[REAL PARAMETER TABLE] Processing {len(rows)-1} data rows from HTML")
            
            for idx, row in enumerate(rows[1:], 1):  # Skip header row
                cells = row.find_all('td')
                if len(cells) < 5:
                    continue
                    
                temp = float(cells[0].text.strip())
                topp = float(cells[1].text.strip())
                topk = int(cells[2].text.strip())
                bm25 = float(cells[3].text.strip())
                
                # Parse concepts from the last cell
                concepts_cell = cells[4]
                concepts = []
                concept_boxes = concepts_cell.find_all('span', class_='concept-box')
                
                for box in concept_boxes:
                    concept_name = box.text.strip()
                    color = box.get('style', '').split('background-color: ')[1].split(';')[0] if 'background-color:' in box.get('style', '') else '#000000'
                    concepts.append((concept_name, color))
                
                print(f"[REAL PARAMETER TABLE] Row {idx}: T={temp}, P={topp}, K={topk}, B={bm25}, Concepts={len(concepts)}")
                
                # Group concepts by folder
                folder_concepts = {}
                for concept, color in concepts:
                    folder = color_to_folder.get(color, 'Unknown')
                    if folder not in folder_concepts:
                        folder_concepts[folder] = set()
                    folder_concepts[folder].add(concept)
                
                print(f"[REAL PARAMETER TABLE] Folders found: {list(folder_concepts.keys())}")
                
                # Calculate metrics for this parameter combination
                param_key = f"T{temp}_P{topp}_K{topk}_B{bm25}"
                parameter_stats[param_key] = {
                    'params': {'temp': temp, 'topp': topp, 'topk': topk, 'bm25': bm25},
                    'folder_concepts': folder_concepts,
                    'total_concepts': len(concepts)
                }
            
            print(f"[REAL PARAMETER TABLE] Created {len(parameter_stats)} parameter combinations")
            
            # Create real table with actual data
            self._create_real_data_table(parent_cell, parameter_stats, first_folder, first_folder_full)
            
        except Exception as e:
            print(f"[REAL PARAMETER TABLE ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    def _create_real_data_table(self, parent_cell, parameter_stats, first_folder, first_folder_full):
        """Create table with real calculated data for first folder"""
        try:
            print(f"[REAL DATA TABLE] Creating table for folder: {first_folder}")
            
            # Create table with 4 columns: Parameter, F Total, F Unique, F Shared
            real_table = parent_cell.add_table(rows=1, cols=4)
            real_table.style = 'Table Grid'
            
            # Add headers
            hdr_cells = real_table.rows[0].cells
            hdr_cells[0].text = "Parameter"
            hdr_cells[1].text = f"{first_folder}\nF Total"
            hdr_cells[2].text = f"{first_folder}\nF Unique"
            hdr_cells[3].text = f"{first_folder}\nF Shared"
            
            # Calculate and add data for each parameter
            parameter_rows = [
                ('Temperature', 'temp'),
                ('Topp', 'topp'), 
                ('Topk', 'topk'),
                ('BM25', 'bm25')
            ]
            
            for param_name, param_key in parameter_rows:
                print(f"[REAL DATA TABLE] Calculating metrics for {param_name}")
                
                # Get all parameter combinations for this parameter type
                relevant_combinations = []
                for param_key_full, stats in parameter_stats.items():
                    if param_key in stats['params']:
                        relevant_combinations.append(stats)
                
                print(f"[REAL DATA TABLE] Found {len(relevant_combinations)} combinations for {param_name}")
                
                # Calculate metrics for this parameter and first folder
                folder_concepts = set()
                for combo in relevant_combinations:
                    # Check both the folder name and full path
                    if first_folder in combo['folder_concepts'] or first_folder_full in combo['folder_concepts']:
                        folder_key = first_folder if first_folder in combo['folder_concepts'] else first_folder_full
                        folder_concepts.update(combo['folder_concepts'][folder_key])
                
                # F Total - Number of concepts for this parameter in this folder
                f_total = len(folder_concepts)
                
                # F Unique - Concepts unique to this parameter in this folder
                f_unique = 0
                for concept in folder_concepts:
                    concept_count = 0
                    for combo in relevant_combinations:
                        # Check both folder name and full path
                        if first_folder in combo['folder_concepts'] and concept in combo['folder_concepts'][first_folder]:
                            concept_count += 1
                        elif first_folder_full in combo['folder_concepts'] and concept in combo['folder_concepts'][first_folder_full]:
                            concept_count += 1
                    if concept_count == 1:
                        f_unique += 1
                
                # F Shared - Concepts shared by this parameter in this folder
                f_shared = f_total - f_unique
                
                print(f"[REAL DATA TABLE] {param_name}: Total={f_total}, Unique={f_unique}, Shared={f_shared}")
                
                # Add row to table
                row_cells = real_table.add_row().cells
                row_cells[0].text = param_name
                row_cells[1].text = str(f_total)
                row_cells[2].text = str(f_unique)
                row_cells[3].text = str(f_shared)
            
            print("[REAL DATA TABLE] Real data table created successfully")
            
        except Exception as e:
            print(f"[REAL DATA TABLE ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    def _create_fallback_parameter_table(self, parent_cell, valid_folders):
        """Create a simple fallback table when data parsing fails"""
        try:
            # Create a simple 3x3 table as fallback
            fallback_table = parent_cell.add_table(rows=3, cols=3)
            fallback_table.style = 'Table Grid'
            
            # Header row
            header_row = fallback_table.rows[0]
            header_row.cells[0].text = "Parameter"
            header_row.cells[0].paragraphs[0].runs[0].bold = True
            header_row.cells[1].text = "Status"
            header_row.cells[1].paragraphs[0].runs[0].bold = True
            header_row.cells[2].text = "Note"
            header_row.cells[2].paragraphs[0].runs[0].bold = True
            
            # Data rows
            parameters = ['Temperature', 'Topp', 'Topk', 'BM25']
            for i, param in enumerate(parameters[:3]):  # Only show first 3
                row = fallback_table.rows[i + 1]
                row.cells[0].text = param
                row.cells[1].text = "Data not available"
                row.cells[2].text = "CSV parsing failed"
            
            print("[PARAMETER COMPARISON] Created fallback table")
            
        except Exception as e:
            print(f"[FALLBACK TABLE ERROR] {e}")

    def _create_parameter_summary_table(self, parent_cell, parameter_stats, valid_folders, color_to_folder):
        """Create the actual parameter summary table"""
        try:
            print(f"[PARAMETER TABLE] Creating table with {len(parameter_stats)} parameter combinations")
            print(f"[PARAMETER TABLE] Valid folders: {valid_folders}")
            
            # Create a simpler table structure
            # Rows: Temperature, Topp, Topk, BM25
            # Columns: Parameter + Each folder (F Total, F Unique, F Shared) + Overall (A Unique, A Shared)
            
            # Calculate number of columns: 1 (parameter) + 3 metrics per folder + 2 overall metrics
            num_cols = 1 + (len(valid_folders) * 3) + 2
            summary_table = parent_cell.add_table(rows=5, cols=num_cols)  # 4 parameter rows + 1 header
            summary_table.style = 'Table Grid'
            
            # Set column widths
            for row in summary_table.rows:
                for i, cell in enumerate(row.cells):
                    if i == 0:  # Parameter name column
                        cell.width = Inches(1.2)
                    else:
                        cell.width = Inches(0.6)
            
            # Header row
            header_row = summary_table.rows[0]
            
            # Set parameter column header
            para = header_row.cells[0].paragraphs[0]
            run = para.add_run("Parameter")
            run.bold = True
            
            col_idx = 1
            # Add folder-specific headers
            for folder in valid_folders:
                # F Total column
                para = header_row.cells[col_idx].paragraphs[0]
                run = para.add_run(f"{folder}\nF Total")
                run.bold = True
                para.alignment = 1  # Center
                col_idx += 1
                
                # F Unique column
                para = header_row.cells[col_idx].paragraphs[0]
                run = para.add_run(f"{folder}\nF Unique")
                run.bold = True
                para.alignment = 1  # Center
                col_idx += 1
                
                # F Shared column
                para = header_row.cells[col_idx].paragraphs[0]
                run = para.add_run(f"{folder}\nF Shared")
                run.bold = True
                para.alignment = 1  # Center
                col_idx += 1
            
            # Overall metrics columns
            para = header_row.cells[col_idx].paragraphs[0]
            run = para.add_run("Overall\nA Unique")
            run.bold = True
            para.alignment = 1  # Center
            col_idx += 1
            
            para = header_row.cells[col_idx].paragraphs[0]
            run = para.add_run("Overall\nA Shared")
            run.bold = True
            para.alignment = 1  # Center
            
            # Calculate and fill data for each parameter
            parameter_rows = [
                ('Temperature', 'temp'),
                ('Topp', 'topp'), 
                ('Topk', 'topk'),
                ('BM25', 'bm25')
            ]
            
            for row_idx, (param_name, param_key) in enumerate(parameter_rows, 1):
                row = summary_table.rows[row_idx]
                para = row.cells[0].paragraphs[0]
                run = para.add_run(param_name)
                run.bold = True
                
                # Calculate metrics for this parameter
                self._calculate_simple_parameter_metrics(row, param_key, parameter_stats, valid_folders)
                
        except Exception as e:
            print(f"[PARAMETER TABLE ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    def _calculate_simple_parameter_metrics(self, row, param_key, parameter_stats, valid_folders):
        """Calculate simple metrics for a specific parameter"""
        try:
            col_idx = 1
            
            # Get all parameter combinations for this parameter type
            relevant_combinations = []
            for param_key_full, stats in parameter_stats.items():
                if param_key in stats['params']:
                    relevant_combinations.append(stats)
            
            print(f"[PARAMETER METRICS] Found {len(relevant_combinations)} combinations for {param_key}")
            
            # Calculate overall unique and shared concepts across all combinations
            all_concepts = set()
            for combo in relevant_combinations:
                for folder_concepts in combo['folder_concepts'].values():
                    all_concepts.update(folder_concepts)
            
            print(f"[PARAMETER METRICS] Total concepts across all combinations: {len(all_concepts)}")
            
            # For each folder, calculate metrics
            for folder in valid_folders:
                folder_concepts = set()
                for combo in relevant_combinations:
                    if folder in combo['folder_concepts']:
                        folder_concepts.update(combo['folder_concepts'][folder])
                
                # F Total - Number of concepts for this parameter in this folder
                f_total = len(folder_concepts)
                
                # F Unique - Concepts unique to this parameter in this folder
                f_unique = 0
                for concept in folder_concepts:
                    concept_count = sum(1 for combo in relevant_combinations 
                                      if folder in combo['folder_concepts'] and concept in combo['folder_concepts'][folder])
                    if concept_count == 1:
                        f_unique += 1
                
                # F Shared - Concepts shared by this parameter in this folder
                f_shared = f_total - f_unique
                
                # Fill the cells
                para = row.cells[col_idx].paragraphs[0]
                para.add_run(str(f_total))
                col_idx += 1
                para = row.cells[col_idx].paragraphs[0]
                para.add_run(str(f_unique))
                col_idx += 1
                para = row.cells[col_idx].paragraphs[0]
                para.add_run(str(f_shared))
                col_idx += 1
                
                print(f"[PARAMETER METRICS] {folder}: Total={f_total}, Unique={f_unique}, Shared={f_shared}")
            
            # Calculate overall A metrics (across all folders)
            a_unique = 0
            a_shared = 0
            for concept in all_concepts:
                concept_count = sum(1 for combo in relevant_combinations 
                                  for folder_concepts in combo['folder_concepts'].values() 
                                  if concept in folder_concepts)
                if concept_count == 1:
                    a_unique += 1
                else:
                    a_shared += 1
            
            # Fill overall metrics
            row.cells[col_idx].text = str(a_unique)
            col_idx += 1
            row.cells[col_idx].text = str(a_shared)
            
            print(f"[PARAMETER METRICS] Overall: Unique={a_unique}, Shared={a_shared}")
            
        except Exception as e:
            print(f"[PARAMETER METRICS ERROR] {e}")
            import traceback
            traceback.print_exc()

    def _calculate_parameter_metrics(self, row, param_name, param_values, parameter_stats, valid_folders, color_to_folder):
        """Calculate metrics for a specific parameter"""
        try:
            col_idx = 1
            
            # Get all parameter combinations for this parameter type
            relevant_combinations = []
            for param_key, stats in parameter_stats.items():
                if param_name.lower() == 'temperature' and stats['params']['temp'] in param_values:
                    relevant_combinations.append(stats)
                elif param_name.lower() == 'topp' and stats['params']['topp'] in param_values:
                    relevant_combinations.append(stats)
                elif param_name.lower() == 'topk' and stats['params']['topk'] in param_values:
                    relevant_combinations.append(stats)
                elif param_name.lower() == 'bm25' and stats['params']['bm25'] in param_values:
                    relevant_combinations.append(stats)
            
            # Calculate overall unique and shared concepts across all combinations
            all_concepts = set()
            for combo in relevant_combinations:
                for folder_concepts in combo['folder_concepts'].values():
                    all_concepts.update(folder_concepts)
            
            # For each folder, calculate metrics
            for folder in valid_folders:
                folder_concepts = set()
                for combo in relevant_combinations:
                    if folder in combo['folder_concepts']:
                        folder_concepts.update(combo['folder_concepts'][folder])
                
                # F Total - Number of concepts for this parameter in this folder
                f_total = len(folder_concepts)
                
                # F Unique - Concepts unique to this parameter in this folder
                f_unique = 0
                for concept in folder_concepts:
                    concept_count = sum(1 for combo in relevant_combinations 
                                      if folder in combo['folder_concepts'] and concept in combo['folder_concepts'][folder])
                    if concept_count == 1:
                        f_unique += 1
                
                # F Shared - Concepts shared by this parameter in this folder
                f_shared = f_total - f_unique
                
                # F Unique% and F Shared% - Percentages from all concepts across all parameters
                total_concepts_all_params = len(all_concepts)
                f_unique_pct = (f_unique / total_concepts_all_params * 100) if total_concepts_all_params > 0 else 0
                f_shared_pct = (f_shared / total_concepts_all_params * 100) if total_concepts_all_params > 0 else 0
                
                # Fill the cells
                row.cells[col_idx].text = str(f_total)
                col_idx += 1
                row.cells[col_idx].text = str(f_unique)
                col_idx += 1
                row.cells[col_idx].text = str(f_shared)
                col_idx += 1
                row.cells[col_idx].text = f"{f_unique_pct:.1f}%"
                col_idx += 1
                row.cells[col_idx].text = f"{f_shared_pct:.1f}%"
                col_idx += 1
                
                # Add empty cells for A metrics (will be filled later)
                for _ in range(3):
                    row.cells[col_idx].text = ""
                    col_idx += 1
            
            # Calculate overall A metrics (across all folders)
            a_unique = 0
            a_shared = 0
            for concept in all_concepts:
                concept_count = sum(1 for combo in relevant_combinations 
                                  for folder_concepts in combo['folder_concepts'].values() 
                                  if concept in folder_concepts)
                if concept_count == 1:
                    a_unique += 1
                else:
                    a_shared += 1
            
            a_unique_pct = (a_unique / len(all_concepts) * 100) if all_concepts else 0
            a_shared_pct = (a_shared / len(all_concepts) * 100) if all_concepts else 0
            
            # Fill overall metrics
            para = row.cells[col_idx].paragraphs[0]
            para.add_run(str(a_unique))
            col_idx += 1
            para = row.cells[col_idx].paragraphs[0]
            para.add_run(str(a_shared))
            col_idx += 1
            row.cells[col_idx].text = f"{a_unique_pct:.1f}%"
            col_idx += 1
            row.cells[col_idx].text = f"{a_shared_pct:.1f}%"
            
        except Exception as e:
            print(f"[PARAMETER METRICS ERROR] {e}")
            import traceback
            traceback.print_exc()

    def _aggregate_results_thread(self):
        import os
        import docx
        from docx.shared import RGBColor, Inches
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from collections import defaultdict, Counter
        import re
        from tkinter import messagebox
        from PIL import Image as PILImage
        import difflib
        import traceback
        import time
        import string
        def safe_update_progress(current_step, total_steps, elapsed, est_total):
            progress_percent = (current_step / total_steps) * 100 if total_steps > 0 else 0
            remaining_time = est_total - elapsed if est_total > elapsed else 0
            self.root.after(0, lambda: self._update_aggregate_progress(progress_percent, elapsed, est_total, current_step, total_steps, remaining_time))
        def safe_update_status(msg, color):
            self.root.after(0, lambda: self.aggregate_status_label.config(text=msg, foreground=color))
        def safe_update_time(msg):
            self.root.after(0, lambda: self.aggregate_time_label.config(text=msg))
        
        # Function to determine if a group is present in a folder
        def is_group_present_in_folder(group_name, concepts, folder_concepts, folder):
            """Check if a group is present in a folder based on the group name, not just any concept in the group"""
            # First, check if any concept that contains the group name appears in the folder
            group_name_lower = group_name.lower()
            for concept in concepts:
                if group_name_lower in concept.lower() and concept in folder_concepts[folder]:
                    return True
            
            # If no concept contains the group name, check if the exact group name appears
            if group_name in folder_concepts[folder]:
                return True
            
            # For single-word group names, be more strict - only return True if the group name
            # appears as a standalone word in a concept, not just as part of another word
            if len(group_name.split()) == 1:
                group_name_lower = group_name.lower()
                for concept in concepts:
                    if concept in folder_concepts[folder]:
                        # Check if the group name appears as a standalone word in the concept
                        words = re.findall(r'\b\w+\b', concept.lower())
                        if group_name_lower in words:
                            return True
            
            return False
        # Define total steps for progress tracking
        total_steps = 8  # Scanning, extracting, grouping, color mapping, DOCX generation, RAG analysis, quotes (optional)
        current_step = 0
        
        safe_update_status("Scanning folders...", "blue")
        current_step += 1
        safe_update_progress(current_step, total_steps, 0, 60)  # Estimate 60 seconds total
        
        parent = self.aggregate_folder
        if not parent or not os.path.isdir(parent):
            safe_update_status("Invalid parent folder.", "red")
            return
        # Get aggregation parameters from GUI
        params = self.get_aggregation_params()
        threshold = params['threshold']
        grouping_logic = params['grouping_logic']
        # Step 1: Find valid subfolders
        subfolders = [os.path.join(parent, d) for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d))]
        valid_folders = []
        for folder in subfolders:
            files = os.listdir(folder)
            required = ["compare_stats.txt", "compare_BM25_composed.png", "compare_Topk_composed.png", "compare_Topp_composed.png", "compare_Temp_composed.png"]
            if all(f in files for f in required):
                valid_folders.append(folder)
        if not valid_folders:
            safe_update_status("No valid result folders found.", "red")
            return
        safe_update_status(f"Found {len(valid_folders)} valid folders. Extracting data...", "blue")
        # Step 2: Extract concepts and images
        current_step += 1
        safe_update_progress(current_step, total_steps, 0, 60)
        
        folder_concepts = {}
        all_concepts = set()
        t0 = time.time()
        for idx, folder in enumerate(valid_folders):
            txt_path = os.path.join(folder, "compare_stats.txt")
            try:
                with open(txt_path, encoding="utf-8") as f:
                    lines = f.readlines()
                # Find 'All Concepts:' section
                concepts = []
                in_concepts = False
                for line in lines:
                    if line.strip().startswith("All Concepts:"):
                        in_concepts = True
                        continue
                    if in_concepts:
                        if not line.strip() or line.strip().endswith(":"):
                            break
                        c = line.strip()
                        if c:
                            concepts.append(c)
                folder_concepts[folder] = set(concepts)
                all_concepts.update(concepts)
            except Exception as e:
                print(f"Error reading {txt_path}: {e}")
            # Progress bar update
            elapsed = time.time() - t0
            if idx == 0 and len(valid_folders) > 1:
                est_total = elapsed * len(valid_folders) * 2  # Estimate 2x for remaining steps
            else:
                est_total = elapsed * 2 if idx == 0 else est_total
            safe_update_progress(current_step, total_steps, elapsed, est_total)
        total_elapsed = time.time() - t0
        # Step 3: Refined Color Grouping Logic
        concept_list = sorted(all_concepts, key=lambda x: x.lower())
        # LLM Grouping logic
        if self.llm_grouping_var.get():
            # Improved prompt for group-name: [concepts] format
            prompt = (
                self.llm_prompt_var.get().strip() +
                "\nReturn a JSON object where each key is a group name and the value is a list of concepts. Example:\n{\n  \"Character\": [\"Character (1103a17–1103b35)\", ...],\n  \"Intellect\": [\"Intellect (Nous)\", ...]\n}"
            )
            model = self.llm_model_var.get()
            print(f"[LLM GROUPING] Starting LLM grouping with model: {model}")
            print(f"[LLM GROUPING] Number of concepts to group: {len(concept_list)}")
            print(f"[LLM GROUPING] Prompt preview: {prompt[:200]}...")
            
            try:
                llm_groups = real_llm_grouping(concept_list, prompt, model, output_dir=parent)
                
                print(f"[LLM GROUPING] LLM grouping completed. Number of groups returned: {len(llm_groups) if llm_groups else 0}")
                if llm_groups:
                    print(f"[LLM GROUPING] First group preview: {llm_groups[0] if llm_groups else 'None'}")
                else:
                    print("[LLM GROUPING] WARNING: No groups returned from LLM!")
                    
            except Exception as e:
                print(f"[LLM GROUPING ERROR] Failed to get LLM response: {e}")
                print(f"[LLM GROUPING ERROR] Exception type: {type(e)}")
                import traceback
                print(f"[LLM GROUPING ERROR] Traceback: {traceback.format_exc()}")
                # Set empty groups and continue with fallback
                llm_groups = []
                llm_error = str(e)
            # Only assign colors to concepts present in LLM output
            concepts_in_llm = set()
            for group in llm_groups:
                for concept in group:
                    concepts_in_llm.add(concept)
            missing_concepts = [c for c in concept_list if c not in concepts_in_llm]
            if missing_concepts:
                msg = f"[LLM GROUPING] {len(missing_concepts)} concepts missing from LLM output. Not assigned to any group.\n" + ", ".join(missing_concepts)
                print(msg)
                self.root.after(0, lambda: self.aggregate_status_label.config(text=msg, foreground="red"))
            color_palette = [
                '#800000', '#FF8C00', '#228B22', '#8B008B', '#A0522D', '#2E8B57', '#9932CC', '#FFD700',
                '#556B2F', '#C71585', '#8B4513', '#20B2AA', '#B22222', '#FF4500', '#6A5ACD', '#D2691E',
                '#006400', '#708090', '#FF6347', '#483D8B', '#000000', '#808000', '#8B0000', '#FF1493',
            ]
            group_colors = {}
            color_to_concepts = defaultdict(list)
            for i, group in enumerate(llm_groups):
                # Use a different color for each group to avoid conflicts
                if i < len(color_palette):
                    color = color_palette[i]
                else:
                    # Generate a unique color for groups beyond the palette
                    import colorsys
                    hue = (i * 0.618) % 1.0  # Golden ratio for good distribution
                    saturation = 0.7
                    value = 0.8
                    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                    color = f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
                
                for concept in group:
                    group_colors[concept] = color
                    color_to_concepts[color].append(concept)

            # For LLM grouping, preserve group structure and color for DOCX output
            if self.llm_grouping_var.get():
                llm_group_tuples = []
                for i, group in enumerate(llm_groups):
                    # Use a different color for each group to avoid conflicts
                    if i < len(color_palette):
                        color = color_palette[i]
                    else:
                        # Generate a unique color for groups beyond the palette
                        import colorsys
                        hue = (i * 0.618) % 1.0  # Golden ratio for good distribution
                        saturation = 0.7
                        value = 0.8
                        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                        color = f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
                    llm_group_tuples.append((color, group))
        else:
            def create_refined_color_groups(concepts, use_colors, group_by_words, group_by_subletters, min_letters):
                """Create refined color groups based on semantic similarity (no transitive merging)"""
                if not use_colors:
                    return {c: '#000000' for c in concepts}
                stopwords = set(['the','a','an','and','or','but','in','on','at','to','for','of','with','by','is','are','was','were','be','been','being','have','has','had','do','does','did','will','would','should','could','can','may','might','must'])
                def extract_words(concept):
                    words = re.findall(r'\b\w+\b', concept.lower())
                    return [w for w in words if w not in stopwords and len(w) >= min_letters]
                # Build word-to-concept mapping
                word_to_concepts = {}
                for concept in concepts:
                    words = extract_words(concept)
                    for w in words:
                        word_to_concepts.setdefault(w, set()).add(concept)
                    # Debug: Print words for Character and Reason concepts
                    if 'Character' in concept or 'Reason' in concept:
                        print(f"[DEBUG] Concept '{concept}' -> words: {words}")
                
                # Build groups: each group is all concepts sharing a word
                groups = []
                assigned = set()
                for w, cset in word_to_concepts.items():
                    group = set(cset) - assigned
                    if len(group) > 1:
                        groups.append(group)
                        assigned.update(group)
                        # Debug: Print groups that contain Character or Reason concepts
                        if any('Character' in str(concept) or 'Reason' in str(concept) for concept in group):
                            print(f"[DEBUG] Group by word '{w}' (len={len(w)}): {sorted(group)}")
                # Any unassigned concepts become their own group
                for concept in concepts:
                    if concept not in assigned:
                        groups.append({concept})
                # Assign colors
                color_palette = [
                    '#800000', '#FF8C00', '#228B22', '#8B008B', '#A0522D', '#2E8B57', '#9932CC', '#FFD700',
                    '#556B2F', '#C71585', '#8B4513', '#20B2AA', '#B22222', '#FF4500', '#6A5ACD', '#D2691E',
                    '#006400', '#708090', '#FF6347', '#483D8B', '#000000', '#808000', '#8B0000', '#FF1493',
                ]
                group_colors = {}
                for i, group in enumerate(groups):
                    # Use a different color for each group to avoid conflicts
                    if i < len(color_palette):
                        color = color_palette[i]
                    else:
                        # Generate a unique color for groups beyond the palette
                        import colorsys
                        hue = (i * 0.618) % 1.0  # Golden ratio for good distribution
                        saturation = 0.7
                        value = 0.8
                        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                        color = f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
                    
                    for concept in group:
                        group_colors[concept] = color
                    # Debug: Print color assignment for groups containing Character or Reason
                    if any('Character' in str(concept) or 'Reason' in str(concept) for concept in group):
                        print(f"[DEBUG] Group {i} assigned color {color}: {sorted(group)}")
                return group_colors
            group_colors = create_refined_color_groups(
                concept_list, 
                params['use_colors'],
                params['group_by_words'],
                params['group_by_subletters'],
                params['min_letters']
            )
            
            # Debug: Print final color assignments for Character and Reason concepts
            print("[DEBUG] Final color assignments:")
            for concept in concept_list:
                if 'Character' in concept or 'Reason' in concept:
                    print(f"  '{concept}' -> {group_colors.get(concept, 'NOT_FOUND')}")
            # --- Fuzzy/Exact/None grouping for overlap and canonicalization ---
            if self.agg_enable_fuzzy.get() and grouping_logic == "Fuzzy":
                groups = []  # List of sets
                used = set()
                for c in concept_list:
                    if c in used:
                        continue
                    group = set([c])
                    for other in concept_list:
                        if other == c or other in used:
                            continue
                        ratio = difflib.SequenceMatcher(None, c.lower(), other.lower()).ratio()
                        if ratio >= threshold:
                            group.add(other)
                            used.add(other)
                    used.add(c)
                    groups.append(group)
            elif grouping_logic == "Exact":
                groups = [{c} for c in concept_list]
            else:  # None or fuzzy disabled
                groups = [{c} for c in concept_list]
            
            # Map each concept to its canonical group label (first in sorted group)
            group_labels = {}
            group_variants = {}
            for group in groups:
                label = sorted(group, key=lambda x: x.lower())[0]
                for variant in group:
                    group_labels[variant] = label
                group_variants[label] = sorted(group, key=lambda x: x.lower())
            canonical_concepts = sorted(group_variants.keys(), key=lambda x: x.lower())
            
            # Step 4: Compute overlap/uniqueness using color groups (unique semantic groups)
            color_to_concepts = defaultdict(list)
            for c in concept_list:
                color = group_colors.get(c, '#FF0000')
                if c not in group_colors:
                    print(f"[GROUP COLOR WARNING] Concept not in group_colors: {c}")
                    self.root.after(0, lambda c=c: self.aggregate_status_label.config(text=f"[GROUP COLOR WARNING] Concept not in group_colors: {c}", foreground='red'))
                    group_colors[c] = '#FF0000'
                color_to_concepts[color].append(c)
        # else: color_to_concepts is already correct from LLM grouping
        
        # Create unique color groups (each color represents a unique semantic group)
        unique_color_groups = list(color_to_concepts.keys())
        
        # Compute overlap using color groups instead of canonical concepts
        color_overlap_table = []  # List of (color, [present in folder1, folder2, ...])
        folder_names = [os.path.basename(f) for f in valid_folders]
        
        if self.llm_grouping_var.get():
            # For LLM grouping, create overlap table from llm_group_tuples
            for color, concepts in llm_group_tuples:
                row = [color]
                concepts_in_color = set(concepts)
                for folder in valid_folders:
                    present = any(c in folder_concepts[folder] for c in concepts_in_color)
                    row.append(1 if present else 0)
                color_overlap_table.append(row)
        else:
            # For color grouping, create overlap table from color_to_concepts
            for color in unique_color_groups:
                row = [color]
                concepts_in_color = set(color_to_concepts[color])
                for folder in valid_folders:
                    present = any(c in folder_concepts[folder] for c in concepts_in_color)
                    row.append(1 if present else 0)
                color_overlap_table.append(row)
        
        # Keep the original canonical overlap table for the final table
        if not self.llm_grouping_var.get():
            overlap_table = []  # List of (canonical, [present in folder1, folder2, ...])
            for canon in canonical_concepts:
                row = [canon]
                canon_variants = set(group_variants[canon])
                for folder in valid_folders:
                    present = any(v in folder_concepts[folder] for v in canon_variants)
                    row.append(1 if present else 0)
                overlap_table.append(row)
        # --- Unique words in folder names ---
        def get_unique_folder_words(folder_names):
            all_words = [set(re.findall(r'\w+', name.lower())) for name in folder_names]
            common = set.intersection(*all_words) if all_words else set()
            unique_words = []
            for words in all_words:
                unique = words - common
                unique_words.append(unique)
            return unique_words, common
        unique_folder_words, common_folder_words = get_unique_folder_words(folder_names)
        folder_unique_map = {folder_names[i]: unique_folder_words[i] for i in range(len(folder_names))}
        def get_concept_folders(concept, folder_concepts, folder_names, valid_folders):
            folders = set()
            for i, folder in enumerate(valid_folders):
                if concept in folder_concepts[folder]:
                    folders.add(folder_names[i])
            return folders
        def get_concept_unique_words(concept, folder_concepts, folder_names, valid_folders, folder_unique_map):
            folders = get_concept_folders(concept, folder_concepts, folder_names, valid_folders)
            unique_words = set()
            for fname in folders:
                unique_words |= folder_unique_map[fname]
            return unique_words
        # Step 5: Collect images
        folder_images = []  # List of (folder_name, [img1, img2, img3, img4])
        for folder in valid_folders:
            imgs = [os.path.join(folder, f) for f in ["compare_BM25_composed.png", "compare_Topk_composed.png", "compare_Topp_composed.png", "compare_Temp_composed.png"]]
            folder_images.append((os.path.basename(folder), imgs))
        # Step 6: Generate DOCX
        current_step += 1
        safe_update_progress(current_step, total_steps, total_elapsed, est_total)
        safe_update_status("Generating DOCX document...", "blue")
        
        try:
            doc = docx.Document()
            # Set landscape orientation and zero margins
            section = doc.sections[0]
            from docx.enum.section import WD_ORIENT
            section.orientation = WD_ORIENT.LANDSCAPE
            section.page_width, section.page_height = section.page_height, section.page_width
            section.top_margin = 0
            section.bottom_margin = 0
            section.left_margin = 0
            section.right_margin = 0
            section.header_distance = 0
            section.footer_distance = 0

            # ===== FIRST PAGE: TOTAL CONCEPTS SUMMARY =====

            # Gather folder/group stats for the left column text
            total_concepts = sum(len(group) for _, group in llm_group_tuples) if self.llm_grouping_var.get() else sum(len(concepts) for concepts in color_to_concepts.values())
            
            # Calculate original concepts counts from folders (before regrouping)
            original_folder_concept_counts = []
            for folder in valid_folders:
                original_folder_concept_counts.append(len(folder_concepts[folder]))
            
            folder_concept_counts = []
            folder_group_counts = []
            for folder in valid_folders:
                concepts_in_folder = set()
                groups_in_folder = 0
                if self.llm_grouping_var.get():
                    for _, group in llm_group_tuples:
                        if any(concept in folder_concepts[folder] for concept in group):
                            groups_in_folder += 1
                            concepts_in_folder.update([concept for concept in group if concept in folder_concepts[folder]])
                else:
                    for color, concepts in color_to_concepts.items():
                        if any(concept in folder_concepts[folder] for concept in concepts):
                            groups_in_folder += 1
                            concepts_in_folder.update([concept for concept in concepts if concept in folder_concepts[folder]])
                folder_concept_counts.append(len(concepts_in_folder))
                folder_group_counts.append(groups_in_folder)

            # Regrouping method
            if self.llm_grouping_var.get():
                method_str = "color grouping based on LLM"
            else:
                if params['group_by_words']:
                    method_str = "color grouping by words"
                elif params['group_by_subletters']:
                    method_str = "color grouping by subletters"
                elif self.agg_enable_fuzzy.get() and params['grouping_logic'] == "Fuzzy":
                    method_str = f"fuzzy grouping (threshold {params['threshold']})"
                else:
                    method_str = "exact grouping"
            num_groups = len(llm_group_tuples) if self.llm_grouping_var.get() else len(color_to_concepts)
            total_unique_groups = len(unique_color_groups)
            
            # Function to extract group name from concepts
            def extract_group_name(concepts):
                """Extract a single common word that represents the group"""
                if not concepts:
                    return "Unknown"
                
                # Try to find the most common meaningful word
                all_words = []
                for concept in concepts:
                    words = re.findall(r'\b\w+\b', concept.lower())
                    # Filter out common words and short words
                    meaningful_words = [w for w in words if len(w) > 3 and w not in ['the', 'and', 'for', 'with', 'from', 'that', 'this', 'have', 'been', 'they', 'will', 'would', 'could', 'should']]
                    all_words.extend(meaningful_words)
                
                if all_words:
                    # Count word frequencies
                    from collections import Counter
                    word_counts = Counter(all_words)
                    # Return the most common word
                    return word_counts.most_common(1)[0][0].capitalize()
                
                # Fallback: use first word of first concept
                first_concept = concepts[0]
                words = re.findall(r'\b\w+\b', first_concept)
                if words:
                    return words[0].capitalize()
            
            
            # Create data for UpSet diagram
            group_data = []
            group_names = []
            
            if self.llm_grouping_var.get():
                for color, concepts in llm_group_tuples:
                    group_name = extract_group_name(concepts)
                    group_names.append(group_name)
                    
                    # Create row for this group: [group_name, present_in_folder1, present_in_folder2, ...]
                    row = [group_name]
                    concepts_in_color = set(concepts)
                    for folder in valid_folders:
                        present = is_group_present_in_folder(group_name, concepts, folder_concepts, folder)
                        row.append(1 if present else 0)
                    group_data.append(row)
            else:
                for color in unique_color_groups:
                    concepts = color_to_concepts[color]
                    group_name = extract_group_name(concepts)
                    group_names.append(group_name)
                    
                    # Create row for this group
                    row = [group_name]
                    concepts_in_color = set(concepts)
                    for folder in valid_folders:
                        present = is_group_present_in_folder(group_name, concepts, folder_concepts, folder)
                        row.append(1 if present else 0)
                    group_data.append(row)
            
            # Create DataFrame for UpSet
            import pandas as pd
            from upsetplot import UpSet, from_indicators
            import matplotlib.pyplot as plt
            
            # Create DataFrame with group names as COLUMNS (not index) - like the existing code
            # This matches the pattern: concept_matrix_reset with concepts as columns
            df_groups = pd.DataFrame(group_data, columns=['Group'] + folder_names)
            
            # Ensure unique group names by adding index if duplicates exist
            unique_group_names = []
            group_name_counts = {}
            for group_name in df_groups['Group']:
                if group_name in group_name_counts:
                    group_name_counts[group_name] += 1
                    unique_name = f"{group_name}_{group_name_counts[group_name]}"
                else:
                    group_name_counts[group_name] = 0
                    unique_name = group_name
                unique_group_names.append(unique_name)
            
            df_groups['Group'] = unique_group_names
            
            # Create the final DataFrame for UpSet plot
            # We need: rows = folders, columns = groups, values = boolean presence
            # Extract the boolean data (folder columns) and use group names as column headers
            group_columns = df_groups[folder_names].T  # Transpose to get groups as columns
            group_columns.columns = unique_group_names  # Set group names as column headers
            
            # Convert to boolean and ensure proper index alignment for UpSet plot
            df_for_upset = group_columns.astype(bool)
            
            # Reset index to ensure clean integer index for UpSet plot
            df_for_upset = df_for_upset.reset_index(drop=True)
            
            # Ensure all values are boolean and handle any NaN values
            df_for_upset = df_for_upset.fillna(False).astype(bool)
            
            # Debug: Print the structure to understand what we have
            print(f"[UPSET DEBUG] df_groups shape: {df_groups.shape}")
            print(f"[UPSET DEBUG] df_groups columns: {df_groups.columns.tolist()}")
            print(f"[UPSET DEBUG] df_groups head:")
            print(df_groups.head())
            print(f"[UPSET DEBUG] group_columns shape after transpose: {group_columns.shape}")
            print(f"[UPSET DEBUG] group_columns columns: {group_columns.columns.tolist()}")
            print(f"[UPSET DEBUG] group_columns head:")
            print(group_columns.head())
            
            print(f"[UPSET DEBUG] Final DataFrame shape: {df_for_upset.shape}")
            print(f"[UPSET DEBUG] Final DataFrame columns: {df_for_upset.columns.tolist()}")
            print(f"[UPSET DEBUG] Final DataFrame dtypes: {df_for_upset.dtypes}")
            print(f"[UPSET DEBUG] Final DataFrame index: {df_for_upset.index}")
            print(f"[UPSET DEBUG] Final DataFrame head:")
            print(df_for_upset.head())
            print(f"[UPSET DEBUG] Final DataFrame info:")
            print(df_for_upset.info())
                        
            upset_plot_path = ""
             # Create UpSet plot with error handling
            try:
                # Use the same pattern as the existing code
                print(f"[UPSET DEBUG] Attempting to create UpSet data with shape: {df_for_upset.shape}")
                print(f"[UPSET DEBUG] DataFrame columns: {list(df_for_upset.columns)}")
                print(f"[UPSET DEBUG] DataFrame index type: {type(df_for_upset.index)}")
                print(f"[UPSET DEBUG] DataFrame index values: {list(df_for_upset.index)}")
                
                upset_data = from_indicators(df_for_upset, df_for_upset.columns)
                print('[UPSET DEBUG] UpSet data created successfully')
                
                # Create figure that fits page height (landscape orientation)
                # Calculate available height for the plot
                page_height_inches = 8.5  # Landscape page height
                margin_inches = 1.0  # Top and bottom margins
                available_height = page_height_inches - (2 * margin_inches)
                
                fig, axes = plt.subplots(1, 1, figsize=(12, available_height))
                
                upset = UpSet(upset_data, show_counts=True)
                axes = upset.plot(fig=fig)
                bar_ax = axes['intersections']
                matrix_ax = axes['matrix']
                bars = bar_ax.patches
                upset_index = upset_data.index
                
                # Add group colors to the y-axis labels (group names)
                yticks = matrix_ax.get_yticklabels()
                for label, group_name in zip(matrix_ax.get_yticklabels(), df_for_upset.columns):
                    # Find the color for this group
                    group_color = None
                    if self.llm_grouping_var.get():
                        # For LLM grouping, find the color from llm_group_tuples
                        for color, concepts in llm_group_tuples:
                            if extract_group_name(concepts) == group_name:
                                group_color = color
                                break
                    else:
                        # For color grouping, find the color from color_to_concepts
                        for color, concepts in color_to_concepts.items():
                            if extract_group_name(concepts) == group_name:
                                group_color = color
                                break
                    
                    if group_color and group_color.startswith('#') and len(group_color) == 7:
                        label.set_color(group_color)
                    else:
                        label.set_color('black')
                    label.set_weight('bold')
                    label.set_fontsize(10)
                
                # Add short folder names as red column labels (rotated 90 degrees)
                # Fixed logic: Skip first bar but ensure labels match correct intersections
                
                # First, collect all non-empty bars with their indices
                non_empty_bars = []
                for i, bar in enumerate(bars):
                    if bar.get_height() > 0:
                        non_empty_bars.append((i, bar))
                
                print(f"[UPSET DEBUG] Total bars: {len(bars)}, Non-empty bars: {len(non_empty_bars)}")
                
                # Create a list of all folder names to cycle through (same as before)
                all_folder_names = []
                for folder in folder_names:
                    folder_unique_words = folder_unique_map.get(folder, set())
                    if folder_unique_words:
                        short_name = sorted(folder_unique_words)[0].capitalize()
                    else:
                        short_name = folder.split()[0].capitalize()
                    all_folder_names.append(short_name)
                
                print(f"[UPSET DEBUG] All folder names for cycling: {all_folder_names}")
                
                # Process all non-empty bars except the first one (skip index 0)
                print(f"[UPSET DEBUG] Non-empty bars indices: {[b[0] for b in non_empty_bars]}")
                
                # Filter out the first bar (index 0) and process the rest
                bars_to_process = [(idx, bar) for idx, bar in non_empty_bars if idx != 0]
                print(f"[UPSET DEBUG] Bars to process: {[b[0] for b in bars_to_process]}")
                

                
                # Fix: Properly map upset plot bars to folders based on actual data content
                # Each red label should represent the folder that best matches the intersection pattern
                
                # Create a method to find the best folder match for each intersection pattern
                def find_best_folder_match(intersection_pattern, df_for_upset, folder_names):
                    """
                    Find which folder best matches the given intersection pattern.
                    Returns the folder index that has the highest overlap with the intersection.
                    """
                    best_match = -1
                    best_score = -1
                    
                    # For each folder (row in df_for_upset), calculate overlap score
                    for folder_idx in range(len(folder_names)):
                        folder_row = df_for_upset.iloc[folder_idx]
                        
                        # Calculate how many concept groups match between intersection and folder
                        matches = 0
                        total_concepts = 0
                        
                        for concept_idx, is_present in enumerate(intersection_pattern):
                            if concept_idx < len(folder_row):
                                if is_present and folder_row.iloc[concept_idx]:
                                    matches += 1
                                if is_present:
                                    total_concepts += 1
                        
                        # Score based on percentage of intersection concepts that are in this folder
                        if total_concepts > 0:
                            score = matches / total_concepts
                            if score > best_score:
                                best_score = score
                                best_match = folder_idx
                    
                    return best_match, best_score
                
                # Assign labels based on best folder matches
                folder_label_assignments = {}
                used_folders = set()
                
                # Process each bar and find its best folder match
                for bar_idx, (original_index, bar) in enumerate(bars_to_process):
                    x = bar.get_x() + bar.get_width() / 2
                    
                    # Get the actual intersection pattern for this bar
                    actual_intersection = upset_index[original_index]
                    
                    # Find the best matching folder for this intersection
                    best_folder_idx, match_score = find_best_folder_match(actual_intersection, df_for_upset, folder_names)
                    
                    if best_folder_idx != -1 and best_folder_idx not in used_folders:
                        # Assign this folder to this bar position
                        folder_label_assignments[bar_idx] = (best_folder_idx, x, original_index, match_score)
                        used_folders.add(best_folder_idx)
                        print(f"[UPSET DEBUG] Bar {bar_idx}: Assigned folder {all_folder_names[best_folder_idx]} "
                              f"(match score: {match_score:.2f})")
                    else:
                        # If no good match or folder already used, find next best available folder
                        for folder_idx in range(len(folder_names)):
                            if folder_idx not in used_folders:
                                folder_label_assignments[bar_idx] = (folder_idx, x, original_index, 0.0)
                                used_folders.add(folder_idx)
                                print(f"[UPSET DEBUG] Bar {bar_idx}: Assigned remaining folder {all_folder_names[folder_idx]}")
                                break
                
                # Draw the labels
                for bar_idx, (folder_idx, x, original_index, match_score) in folder_label_assignments.items():
                    label = all_folder_names[folder_idx]
                    y_label = len(df_for_upset.columns) - 0.3
                    
                    # Place the label
                    matrix_ax.text(x, y_label, label, ha='center', va='bottom', 
                                  fontsize=10, color='red', rotation=90, 
                                  clip_on=False, weight='bold')
                    
                    actual_intersection = upset_index[original_index]
                    present_folders = []
                    for j, present in enumerate(actual_intersection):
                        if present and j < len(folder_names):
                            present_folders.append(j)
                    
                    print(f"[UPSET DEBUG] Bar (original index: {original_index}, bar_idx: {bar_idx}): "
                          f"Drawing folder label: '{label}' at x={x:.2f}, "
                          f"intersection: {actual_intersection}, present_folders: {present_folders}, "
                          f"match_score: {match_score:.2f}")
                
                # Customize the plot
                plt.title("Concept Groups Overlap Across Authors", fontsize=14, pad=20)
                
                # Save the plot
                upset_plot_path = os.path.join(parent, "groups_upset_plot.png")
                plt.savefig(upset_plot_path, dpi=150, bbox_inches='tight', pad_inches=0.5)
                plt.close()
                
            except Exception as e:
                print(f"[UPSET ERROR] Failed to create UpSet plot: {e}")
                print(f"[UPSET ERROR] Exception type: {type(e)}")
                import traceback
                print(f"[UPSET ERROR] Traceback: {traceback.format_exc()}")                
                
                # Create a placeholder plot to avoid empty file path error
                try:
                    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                    ax.text(0.5, 0.5, 'UpSet Plot\nCould Not Be Generated', 
                           ha='center', va='center', fontsize=16, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')
                    upset_plot_path = os.path.join(parent, "groups_upset_plot_placeholder.png")
                    plt.savefig(upset_plot_path, dpi=150, bbox_inches='tight', pad_inches=0.5)
                    plt.close()
                    print(f"[UPSET ERROR] Created placeholder plot at: {upset_plot_path}")
                except Exception as placeholder_error:
                    print(f"[UPSET ERROR] Failed to create placeholder plot: {placeholder_error}")
                    upset_plot_path = ""  # Will be handled later
                
                # Add overlap summary text to the first page (before the UpSet diagram)
                
            # Add the UpSet diagram in a table format (summary on left, diagram on right)
            doc.add_heading("Overlap Summary", level=1)
            
            # Add RAG Consistency Analysis Tables
            current_step += 1
            safe_update_progress(current_step, total_steps, total_elapsed, est_total)
            safe_update_status("Running RAG consistency analysis...", "blue")
            self._add_rag_consistency_analysis(doc, valid_folders, folder_concepts, parent)
            
            # Add Quotes Table if enabled (independent of consistency analysis)
            if self.add_quotes_var.get():
                current_step += 1
                safe_update_progress(current_step, total_steps, total_elapsed, est_total)
                safe_update_status("Processing quotes and citations...", "blue")
                
                try:
                    # Read quotes from compare_stats.docx files in each folder
                    print(f"[QUOTES] Starting quotes processing for {len(valid_folders)} folders")
                    quotes_data = self._read_quotes_from_stats_docs(valid_folders)
                    print(f"[QUOTES] Quotes data collected: {len(quotes_data)} concepts")
                    
                    # Check if we have any quotes data
                    total_quotes = 0
                    for concept, folders in quotes_data.items():
                        for folder, data in folders.items():
                            total_quotes += len(data.get('quotes', []))
                    print(f"[QUOTES] Total quotes found: {total_quotes}")
                    
                    if total_quotes > 0:
                        # Add quotes table to document
                        self._add_quotes_table_to_doc(doc, valid_folders, quotes_data, parent)
                        print(f"[QUOTES] Successfully added quotes table to document")
                    else:
                        print(f"[QUOTES] No quotes found, skipping quotes table")
                        doc.add_heading("Concept Citations and Quotes", level=2)
                        doc.add_paragraph("No quotes were found in the compare_stats.docx files for the selected concepts.")
                    
                except Exception as e:
                    print(f"[QUOTES ERROR] Failed to process quotes: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue with the rest of the process even if quotes fail
                    safe_update_status("Quotes processing failed, continuing...", "orange")
            
            
            plot_table = create_fixed_width_table(doc, rows=1, cols=2, col_widths_inches=[6, 9])

            # Left column for overlap summary text
            left_cell = plot_table.rows[0].cells[0]

            # Add overlap summary text to left column
            para = left_cell.paragraphs[0]
            
            # Create a smaller table with original concepts on left and regrouped info on right
            summary_table = left_cell.add_table(rows=len(folder_names) + 1, cols=2)
            summary_table.style = 'Table Grid'
            
            # Set column widths (left: original concepts, right: regrouped info)
            for row in summary_table.rows:
                row.cells[0].width = Inches(2.5)  # Left column for original concepts
                row.cells[1].width = Inches(3.5)  # Right column for regrouped info
            
            # Header row
            header_cell_left = summary_table.rows[0].cells[0]
            header_cell_left.text = "Original Concepts"
            header_cell_left.paragraphs[0].runs[0].bold = True
            
            header_cell_right = summary_table.rows[0].cells[1]
            # Determine regrouping method for display
            if self.llm_grouping_var.get():
                regrouping_method = self.llm_model_var.get()
            else:
                regrouping_method = "common words"
            
            header_cell_right.text = f"After Regrouping using {regrouping_method}"
            header_cell_right.paragraphs[0].runs[0].bold = True
            
            # Data rows
            for i, name in enumerate(folder_names):
                # Left column: Original concepts count
                left_cell_data = summary_table.rows[i + 1].cells[0]
                left_para = left_cell_data.paragraphs[0]
                left_para.add_run(f"📁 {name}: ").bold = True
                left_para.add_run(f"{original_folder_concept_counts[i]} concepts").bold = True
                
                # Right column: Regrouped info
                right_cell_data = summary_table.rows[i + 1].cells[1]
                right_para = right_cell_data.paragraphs[0]
                right_para.add_run(f"📁 {name}: ").bold = True
                right_para.add_run(f"{folder_concept_counts[i]} concepts; {folder_group_counts[i]} groups").bold = True
            
            # Add total concepts information below the table
            para = left_cell.add_paragraph()
            para.add_run("📊 Total concepts: ").bold = True
            run = para.add_run(str(total_concepts))
            run.bold = True
            para.add_run("\n")
            
            # Parameter comparison table removed as requested
            
            # Regrouping method
            para.add_run("🎨 The concepts have been regrouped into ").bold = True
            run = para.add_run(str(num_groups))
            run.bold = True
            para.add_run(" concept groups through the method: ")
            run2 = para.add_run(method_str)
            run2.bold = True
            
            # Add LLM prompt information if LLM grouping is used - all on one line
            if self.llm_grouping_var.get():
                # Extract the instruction part (excluding the concepts list)
                full_prompt = self.llm_prompt_var.get().strip()
                # Remove the part that adds concepts list - be more flexible with the pattern
                concepts_section = "\nConcepts:\n" + "\n".join(f"- {c}" for c in concept_list)
                if concepts_section in full_prompt:
                    instruction_part = full_prompt.replace(concepts_section, "").strip()
                else:
                    # If the pattern doesn't match exactly, just use the original prompt
                    instruction_part = full_prompt
                
                para.add_run(" 🤖 LLM Model: ").bold = True
                para.add_run(self.llm_model_var.get()).bold = True
                para.add_run(" 📝 LLM Instruction: ").bold = True
                para.add_run(instruction_part)
                
                # Add LLM response debugging information
                if 'llm_groups' in locals() and llm_groups:
                    para.add_run(" 📊 LLM Response: ").bold = True
                    para.add_run(f"Successfully grouped into {len(llm_groups)} groups")
                else:
                    para.add_run(" ❌ LLM Response: ").bold = True
                    para.add_run("No response received from LLM")
                        
            # Add missing concepts information for LLM grouping
            if self.llm_grouping_var.get() and missing_concepts:
                para.add_run("\n\n⚠️ Note: ").bold = True
                run3 = para.add_run(f"{len(missing_concepts)} concepts were not assigned to any group by the LLM and are excluded from the analysis: ")
                run3.bold = True
                para.add_run(", ".join(sorted(missing_concepts)))
                para.add_run("\n\n")
            
            # Add error information if there were any errors during processing
            if 'llm_error' in locals() and llm_error:
                para.add_run("❌ LLM Processing Error: ").bold = True
                para.add_run(str(llm_error))
                para.add_run("\n\n")
            
            # Unique/Shared groups per folder
            for i, folder in enumerate(valid_folders):
                unique_groups = sum(1 for row in color_overlap_table if row[i+1] and sum(row[1:]) == 1)
                shared_groups = sum(1 for row in color_overlap_table if all(row[1:]))
                percent_unique = 100.0 * unique_groups / total_unique_groups if total_unique_groups else 0
                percent_shared = 100.0 * shared_groups / total_unique_groups if total_unique_groups else 0
                
                # Get unique capitalized folder name
                folder_unique_words = folder_unique_map.get(folder_names[i], set())
                if folder_unique_words:
                    short_folder_name = sorted(folder_unique_words)[0].capitalize()
                else:
                    short_folder_name = folder_names[i].split()[0].capitalize()
                
                para.add_run(f"🟢 {short_folder_name}: ").bold = True
                run = para.add_run(f"{unique_groups}")
                run.bold = True
                para.add_run(" unique concept groups (")
                run = para.add_run(f"{percent_unique:.1f}%")
                run.bold = True
                para.add_run("), ")
                run = para.add_run(f"{shared_groups}")
                run.bold = True
                para.add_run(" shared concept groups (")
                run = para.add_run(f"{percent_shared:.1f}%")
                run.bold = True
                para.add_run(")\n")
            
            # Add individual concepts upset plots to left column (below stats)
            try:
                # Function to create upset plot for individual concepts
                def create_individual_upset_plot(concepts_list, title, output_path, fig_width=3, fig_height=4):
                    # Create data for individual concepts upset plot
                    concept_data = []
                    for concept in concepts_list:
                        row = [concept]
                        for folder in valid_folders:
                            present = concept in folder_concepts[folder]
                            row.append(1 if present else 0)
                        concept_data.append(row)
                    
                    df_concepts = pd.DataFrame(concept_data, columns=['Concept'] + folder_names)
                    
                    # Ensure unique concept names for index
                    unique_concept_names = []
                    concept_name_counts = {}
                    for concept in df_concepts['Concept']:
                        if concept in concept_name_counts:
                            concept_name_counts[concept] += 1
                            unique_name = f"{concept}_{concept_name_counts[concept]}"
                        else:
                            concept_name_counts[concept] = 0
                            unique_name = concept
                        unique_concept_names.append(unique_name)
                    df_concepts['Concept'] = unique_concept_names
                    
                    # Set index to concept names, columns to short folder names
                    short_folder_names = [",".join(sorted(folder_unique_map[f])) if folder_unique_map[f] else f for f in folder_names]
                    df_for_upset_individual = df_concepts.set_index('Concept')[folder_names]
                    df_for_upset_individual.columns = short_folder_names
                    df_for_upset_individual = df_for_upset_individual.astype(bool)
                    
                    # Create individual concepts upset plot
                    upset_data_individual = from_indicators(df_for_upset_individual)
                    
                    # Create figure
                    fig_individual_left, axes_individual_left = plt.subplots(1, 1, figsize=(fig_width, fig_height))
                    upset_individual_left = UpSet(upset_data_individual, show_counts=True, subset_size='count')
                    upset_individual_left.plot(fig=fig_individual_left)
                    
                    plt.title(title, fontsize=10, pad=10)
                    plt.tight_layout()
                    
                    # Save the plot
                    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
                    plt.close()
                
                # Function to create upset plot for groups (using group names only)
                def create_group_upset_plot(group_names_list, title, output_path, fig_width=3, fig_height=4):
                    # Create data for groups upset plot
                    group_data_plot = []
                    for group_name in group_names_list:
                        row = [group_name]
                        for folder in valid_folders:
                            # Check if any concept in this group is present in the folder
                            present = False
                            if self.llm_grouping_var.get():
                                for color, concepts in llm_group_tuples:
                                    if extract_group_name(concepts) == group_name:
                                        present = is_group_present_in_folder(group_name, concepts, folder_concepts, folder)
                                        break
                            else:
                                for color, concepts in color_to_concepts.items():
                                    if extract_group_name(concepts) == group_name:
                                        present = is_group_present_in_folder(group_name, concepts, folder_concepts, folder)
                                        break
                            row.append(1 if present else 0)
                        group_data_plot.append(row)
                    
                    df_groups_plot = pd.DataFrame(group_data_plot, columns=['Group'] + folder_names)
                    
                    # Set index to group names, columns to short folder names
                    short_folder_names = [",".join(sorted(folder_unique_map[f])) if folder_unique_map[f] else f for f in folder_names]
                    df_for_upset_groups = df_groups_plot.set_index('Group')[folder_names]
                    df_for_upset_groups.columns = short_folder_names
                    df_for_upset_groups = df_for_upset_groups.astype(bool)
                    
                    # Create groups upset plot
                    upset_data_groups = from_indicators(df_for_upset_groups)
                    
                    # Create figure
                    fig_groups_left, axes_groups_left = plt.subplots(1, 1, figsize=(fig_width, fig_height))
                    upset_groups_left = UpSet(upset_data_groups, show_counts=True, subset_size='count')
                    upset_groups_left.plot(fig=fig_groups_left)
                    
                    plt.title(title, fontsize=10, pad=10)
                    plt.tight_layout()
                    
                    # Save the plot
                    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
                    plt.close()
                
                # Get all individual concepts from color_to_concepts (BEFORE grouping)
                all_individual_concepts = []
                for color, concepts in color_to_concepts.items():
                    all_individual_concepts.extend(concepts)
                
                # Remove duplicates while preserving order
                seen = set()
                unique_concepts_before = []
                for concept in all_individual_concepts:
                    if concept not in seen:
                        seen.add(concept)
                        unique_concepts_before.append(concept)
                
                # Create BEFORE grouping upset plot
                before_title = f"ALL {len(unique_concepts_before)} concepts BEFORE Grouping"
                before_path = os.path.join(parent, "individual_concepts_upset_before.png")
                create_individual_upset_plot(unique_concepts_before, before_title, before_path, fig_width=3, fig_height=3)
                
                # Get group names from Group Presence Matrix (AFTER grouping)
                group_names_after = []
                for i, row_data in enumerate(group_data):
                    group_name = unique_group_names[i] if i < len(unique_group_names) else row_data[0]
                    group_names_after.append(group_name)
                
                # Create AFTER grouping upset plot (using group names only)
                after_title = f"ALL {len(group_names_after)} concepts AFTER Grouping"
                after_path = os.path.join(parent, "group_upset_after.png")
                create_group_upset_plot(group_names_after, after_title, after_path, fig_width=3, fig_height=3)
                

                # Add both plots to a nested table in the left column
                # Create a nested table in the left cell (using python-docx's underlying XML)
                def add_nested_table(cell, rows, cols, total_width_inches=2.8):
                    """
                    Create a nested table with proper column width settings
                    """
                    from docx.oxml import OxmlElement
                    from docx.oxml.ns import qn
                    from docx.table import Table
                    import docx.shared
                    
                    tbl = OxmlElement('w:tbl')
                    
                    # Table properties
                    tblPr = OxmlElement('w:tblPr')
                    
                    # Set table width to a specific value instead of auto
                    tblW = OxmlElement('w:tblW')
                    tblW.set(qn('w:w'), str(int(total_width_inches * 1440)))  # Convert inches to twips (1440 twips per inch)
                    tblW.set(qn('w:type'), 'dxa')  # dxa = twentieths of a point
                    tblPr.append(tblW)
                    
                    # Table layout - fixed instead of auto
                    tblLayout = OxmlElement('w:tblLayout')
                    tblLayout.set(qn('w:type'), 'fixed')
                    tblPr.append(tblLayout)
                    
                    tbl.append(tblPr)
                    
                    # Table grid with specific column widths
                    tblGrid = OxmlElement('w:tblGrid')
                    col_width_twips = int((total_width_inches / cols) * 1440)  # Equal width columns
                    
                    for i in range(cols):
                        gridCol = OxmlElement('w:gridCol')
                        gridCol.set(qn('w:w'), str(col_width_twips))
                        tblGrid.append(gridCol)
                    tbl.append(tblGrid)
                    
                    # Create rows and cells
                    for i in range(rows):
                        tr = OxmlElement('w:tr')
                        for j in range(cols):
                            tc = OxmlElement('w:tc')
                            
                            # Cell properties with specific width
                            tcPr = OxmlElement('w:tcPr')
                            tcW = OxmlElement('w:tcW')
                            tcW.set(qn('w:w'), str(col_width_twips))
                            tcW.set(qn('w:type'), 'dxa')
                            tcPr.append(tcW)
                            tc.append(tcPr)
                            
                            # Add paragraph to cell
                            p = OxmlElement('w:p')
                            tc.append(p)
                            tr.append(tc)
                        tbl.append(tr)
                    
                    # Append to parent cell
                    cell._element.append(tbl)
                    return Table(tbl, cell._parent)

                # Usage (replace your existing code):
                new_para = left_cell.add_paragraph()
    
                # Create nested table in the NEW paragraph
                nested_table = add_nested_table(new_para, 1, 2, total_width_inches=6.5)
                
                # Insert BEFORE plot
                left_cell = nested_table.cell(0, 0)
                left_cell_paragraph = left_cell.paragraphs[0]
                left_cell_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run_before_plot = left_cell_paragraph.add_run()
                run_before_plot.add_picture(before_path, width=Inches(2.5))
                
                # Insert AFTER plot  
                right_cell_nested = nested_table.cell(0, 1)
                right_cell_paragraph = right_cell_nested.paragraphs[0]
                right_cell_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run_after_plot = right_cell_paragraph.add_run()
                run_after_plot.add_picture(after_path, width=Inches(3.5))
                                
            except Exception as e:
                print(f"[LEFT COLUMN INDIVIDUAL UPSET ERROR] {e}")
                # If plot fails, just add a text note
                para.add_run("\n[Individual concepts plots could not be generated]")

            page_height_inches = 8.5  # 8.5 inches * 72 points per inch
            margin_inches = 0.0  # 1 inch margins
            available_height_inches = page_height_inches - (2 * margin_inches)

            # Right column for the plot
            right_cell = plot_table.rows[0].cells[1]
            
            # Cell properties for better control
            right_cell_paragraph = right_cell.paragraphs[0]            
            right_cell_paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            # Clear any existing content
            right_cell_paragraph.clear()

            # Create run for the image
            run = right_cell_paragraph.add_run()
        
            # Cap the height to a reasonable maximum to prevent layout issues
            max_plot_height = min(available_height_inches, 7)  # Max 7 inches

            # Only add the picture if the plot path exists and is not empty
            if upset_plot_path and os.path.exists(upset_plot_path):
                run.add_picture(upset_plot_path, height=Inches(max_plot_height))  # Use most of the right column width
            else:
                # Add a text message if no plot is available
                run.add_text("UpSet plot could not be generated due to data processing error.")

            # Create and add the individual concepts upset plot            
            # This upset plot shows individual concepts (not grouped) and which folders they appear in
            try:
                # Create data for individual concepts upset plot
                concept_data = []
                concept_names = []
                
                # Get all individual concepts from color_to_concepts
                all_individual_concepts = []
                for color, concepts in color_to_concepts.items():
                    all_individual_concepts.extend(concepts)
                
                # Remove duplicates while preserving order
                seen = set()
                unique_concepts = []
                for concept in all_individual_concepts:
                    if concept not in seen:
                        seen.add(concept)
                        unique_concepts.append(concept)
                
                # Create row for each individual concept
                for concept in unique_concepts:
                    concept_names.append(concept)
                    row = [concept]
                    for folder in valid_folders:
                        present = concept in folder_concepts[folder]
                        row.append(1 if present else 0)
                    concept_data.append(row)
                
                # Create DataFrame for individual concepts upset plot
                df_concepts = pd.DataFrame(concept_data, columns=['Concept'] + folder_names)
                
                # Create the final DataFrame with concepts as columns (like the group upset plot)
                concept_columns = df_concepts[folder_names].T  # Transpose to get concepts as columns
                concept_columns.columns = concept_names  # Set concept names as column headers
                
                # Convert to boolean and reset index to get default integer index
                df_concepts_for_upset = concept_columns.astype(bool).reset_index(drop=True)
                
                print(f"[INDIVIDUAL UPSET DEBUG] Final DataFrame shape: {df_concepts_for_upset.shape}")
                print(f"[INDIVIDUAL UPSET DEBUG] Final DataFrame columns: {df_concepts_for_upset.columns.tolist()}")
                
                # Create individual concepts upset plot
                individual_upset_data = from_indicators(df_concepts_for_upset, df_concepts_for_upset.columns)
                print('[INDIVIDUAL UPSET DEBUG] Individual UpSet data created successfully')
                
                # Create figure for individual concepts upset plot
                fig_individual, axes_individual = plt.subplots(1, 1, figsize=(12, max(6, 0.4 * len(df_concepts_for_upset))))
                
                upset_individual = UpSet(individual_upset_data, show_counts=True)
                axes_individual = upset_individual.plot(fig=fig_individual)
                bar_ax_individual = axes_individual['intersections']
                matrix_ax_individual = axes_individual['matrix']
                
                # Add concept colors to the y-axis labels (concept names)
                for label, concept_name in zip(matrix_ax_individual.get_yticklabels(), df_concepts_for_upset.columns):
                    concept_color = group_colors.get(concept_name, '#000000')
                    if concept_color and concept_color.startswith('#') and len(concept_color) == 7:
                        label.set_color(concept_color)
                    else:
                        label.set_color('black')
                    label.set_weight('bold')
                    label.set_fontsize(8)
                
                # Add red column labels based on Folder Unique Words for each concept
                for bar_idx, bar in enumerate(bar_ax_individual.patches):
                    if bar.get_height() == 0:
                        continue
                    x = bar.get_x() + bar.get_width() / 2
                    intersection = individual_upset_data.index[bar_idx]
                    present_folders = []
                    for j, present in enumerate(intersection):
                        if present and j < len(folder_names):
                            concept = df_concepts_for_upset.columns[j]
                            folders = get_concept_folders(concept, folder_concepts, folder_names, valid_folders)
                            unique_words = set()
                            for fname in folders:
                                unique_words |= folder_unique_map[fname]
                            if unique_words:
                                label_txt = ", ".join(sorted(unique_words)).capitalize()
                            else:
                                label_txt = folder_names[j].split()[0].capitalize()
                            present_folders.append(label_txt)
                    if present_folders:
                        matrix_ax_individual.text(x, len(df_concepts_for_upset.columns) - 0.3, ", ".join(present_folders), ha='center', va='bottom', fontsize=8, color='red', rotation=90, clip_on=False, weight='bold')
                
                # Save the individual concepts upset plot
                individual_upset_plot_path = os.path.join(parent, "individual_concepts_upset_plot.png")
                plt.savefig(individual_upset_plot_path, dpi=150, bbox_inches='tight', pad_inches=0.5)
                plt.close()
                                
            except Exception as e:
                print(f"[INDIVIDUAL UPSET ERROR] Failed to create individual concepts UpSet plot: {e}")
                print(f"[INDIVIDUAL UPSET ERROR] Exception type: {type(e)}")
                import traceback
                print(f"[INDIVIDUAL UPSET ERROR] Traceback: {traceback.format_exc()}")

            # --- Individual Concepts UpSet Plot (upset_alt.py style) ---
            # Prepare data: each row is a concept, columns are folders, value is True if concept in folder
            concept_data = []
            for concept in unique_concepts:
                row = [concept]
                for folder in valid_folders:
                    present = concept in folder_concepts[folder]
                    row.append(1 if present else 0)
                concept_data.append(row)

            df_concepts = pd.DataFrame(concept_data, columns=['Concept'] + folder_names)

            # Ensure unique concept names for index
            unique_concept_names = []
            concept_name_counts = {}
            for concept in df_concepts['Concept']:
                if concept in concept_name_counts:
                    concept_name_counts[concept] += 1
                    unique_name = f"{concept}_{concept_name_counts[concept]}"
                else:
                    concept_name_counts[concept] = 0
                    unique_name = concept
                unique_concept_names.append(unique_name)
            df_concepts['Concept'] = unique_concept_names

            # Set index to concept names, columns to short folder names
            short_folder_names = [",".join(sorted(folder_unique_map[f])) if folder_unique_map[f] else f for f in folder_names]
            df_for_upset = df_concepts.set_index('Concept')[folder_names]
            df_for_upset.columns = short_folder_names
            df_for_upset = df_for_upset.astype(bool)

            # Plot - PROPER APPROACH FOR ALL CONCEPTS
            try:
                # With ALL concepts, UpSet plot becomes impractical - use alternative approach
                num_concepts = len(df_for_upset)
                print(f"Processing {num_concepts} concepts...")
                
                if num_concepts > 50:
                    # Calculate concept frequencies for sorting
                    concept_frequencies = df_for_upset.sum(axis=1).sort_values(ascending=False)
                    
                    # Sort concepts by frequency for better visualization
                    df_sorted = df_for_upset.loc[concept_frequencies.index]
                    
                    # Create figure and axis for heatmap
                    fig, ax = plt.subplots(figsize=(12, max(8, 0.3 * num_concepts)))
                    
                    # HEATMAP
                    import seaborn as sns
                    sns.heatmap(df_sorted.astype(int), 
                               annot=False, 
                               cmap='Blues', 
                               cbar=True,
                               xticklabels=True,
                               yticklabels=True,
                               ax=ax,
                               cbar_kws={'label': 'Concept Present'})
                    
                    plt.title(f"All {num_concepts} Concepts Across Authors (Sorted by Frequency)", fontsize=14)
                    plt.xlabel("Folders", fontsize=12)
                    plt.ylabel("Concepts (sorted by frequency)", fontsize=12)
                    
                    # Fix x-axis labels: horizontal, capitalized, from bottom
                    x_labels = [label.get_text().upper() for label in ax.get_xticklabels()]
                    ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=10)
                    
                    # Keep y-axis labels small but readable
                    y_labels = [label.get_text() for label in ax.get_yticklabels()]
                    ax.set_yticklabels(y_labels, rotation=0, fontsize=6)
                    
                    # Position x-axis labels at bottom
                    ax.xaxis.set_ticks_position('bottom')
                    ax.xaxis.set_label_position('bottom')
                    
                    # Adjust layout
                    plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.15)
                    
                    plt.savefig(os.path.join(parent, "all_concepts_heatmap.png"), 
                                dpi=150, bbox_inches='tight', pad_inches=0.5)
                    plt.close()
                    
                    print(f"Created UpSet for all {num_concepts} concepts and heatmap for all {num_concepts} concepts")
                    
                else:
                    # For smaller numbers, use original UpSet approach
                    upset_data = from_indicators(df_for_upset)
                    fig_height = max(8, 0.4 * num_concepts + 4)
                    
                    fig, axes = plt.subplots(1, 1, figsize=(14, fig_height))
                    upset = UpSet(upset_data, show_counts=True, subset_size='count')
                    upset.plot(fig=fig)
                    
                    plt.title("Individual Concepts Overlap Across Authors", fontsize=14, pad=20)
                    plt.tight_layout()
                    plt.savefig(os.path.join(parent, "individual_concepts_upset_alt.png"), 
                                dpi=150, bbox_inches='tight', pad_inches=0.5)
                    plt.close()
                    
                    print(f"Successfully created UpSet plot with {num_concepts} concepts")
                
            except Exception as e:
                print(f"[UPSET_ALT INDIVIDUAL ERROR] {e}")
                print(f"[UPSET_ALT INDIVIDUAL ERROR] Exception type: {type(e)}")
                import traceback
                print(f"[UPSET_ALT INDIVIDUAL ERROR] Traceback: {traceback.format_exc()}")
                print(f"Number of concepts: {len(df_for_upset) if 'df_for_upset' in locals() else 'Unknown'}")
                
                # Fallback: Create a simple alternative visualization
                try:
                    fig, ax = plt.subplots(figsize=(12, max(8, 0.3 * len(df_for_upset))))
                    
                    # Create a heatmap showing all concepts
                    import seaborn as sns
                    sns.heatmap(df_for_upset.astype(int), 
                               annot=False, 
                               cmap='Blues', 
                               cbar=True,
                               xticklabels=True,
                               yticklabels=True,
                               ax=ax)
                    
                    plt.title("Individual Concepts Presence Across Authors (Heatmap Fallback)", fontsize=14)
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0, fontsize=8)
                    plt.tight_layout()
                    plt.savefig(os.path.join(parent, "individual_concepts_heatmap_fallback.png"), 
                                dpi=150, bbox_inches='tight')
                    plt.close()
                    print("Created fallback heatmap visualization")
                    
                except Exception as fallback_error:
                    print(f"[FALLBACK ERROR] {fallback_error}")
                    
                    
            # Add Group Presence Matrix on the second pag
            doc.add_page_break()
            doc.add_heading("Group Presence Matrix", level=2)
            group_table = doc.add_table(rows=1, cols=1+len(folder_names))
            group_table.rows[0].cells[0].text = "Group"
            for i, folder_name in enumerate(folder_names):
                # Get unique capitalized folder name
                folder_unique_words = folder_unique_map.get(folder_name, set())
                if folder_unique_words:
                    short_folder_name = sorted(folder_unique_words)[0].capitalize()
                else:
                    short_folder_name = folder_name.split()[0].capitalize()
                group_table.rows[0].cells[1+i].text = short_folder_name
            
            # Make headings bold and smaller font
            for cell in group_table.rows[0].cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        run.bold = True
                        run.font.size = docx.shared.Pt(9)
            
            # Add data rows with colored concepts and smaller font
            for i, row_data in enumerate(group_data):
                row = group_table.add_row().cells
                # Use the unique group name from the DataFrame
                group_name = unique_group_names[i] if i < len(unique_group_names) else row_data[0]
                row[0].text = group_name
                
                # Set smaller font for all cells in this row
                for cell in row:
                    for para in cell.paragraphs:
                        for run in para.runs:
                            run.font.size = docx.shared.Pt(8)
                
                # Color the group name based on the group's color
                if self.llm_grouping_var.get():
                    # For LLM grouping, find the color from llm_group_tuples
                    for color, concepts in llm_group_tuples:
                        if extract_group_name(concepts) == group_name:
                            if color.startswith('#') and len(color) == 7:
                                r, g, b = tuple(int(color[j:j+2], 16) for j in (1, 3, 5))
                                row[0].paragraphs[0].runs[0].font.color.rgb = RGBColor(r, g, b)
                            break
                else:
                    # For color grouping, find the color from color_to_concepts
                    for color, concepts in color_to_concepts.items():
                        if extract_group_name(concepts) == group_name:
                            if color.startswith('#') and len(color) == 7:
                                r, g, b = tuple(int(color[j:j+2], 16) for j in (1, 3, 5))
                                row[0].paragraphs[0].runs[0].font.color.rgb = RGBColor(r, g, b)
                            break
                
                # Add presence indicators with green/red emoticons
                for j, present in enumerate(row_data[1:]):
                    row[1+j].text = "🟢" if present else "🔴"

            # --- Unique/Common Concepts Table ---
            doc.add_heading("Unique/Common Concepts Table", level=2)
            uniq_table = doc.add_table(rows=1, cols=5)
            widths = [1.5, 4.5, 1.2, 1.5, 2.0]  # inches - adjusted for new Group column
            for i, w in enumerate(widths):
                uniq_table.columns[i].width = docx.shared.Inches(w)
            # Make headings bold
            for cell in uniq_table.rows[0].cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        run.bold = True
            uniq_table.rows[0].cells[0].text = "Group"
            uniq_table.rows[0].cells[1].text = "Concepts"
            uniq_table.rows[0].cells[2].text = "Unique/Shared"
            uniq_table.rows[0].cells[3].text = "Authors"
            uniq_table.rows[0].cells[4].text = "#Concepts"
            
            folder_unique_words_map = {}
            all_concepts_table = {}
            for color, concepts in (llm_group_tuples if self.llm_grouping_var.get() else color_to_concepts.items()):
                for concept in concepts:
                    folders_with_concept = set()
                    for i, folder in enumerate(valid_folders):
                        if concept in folder_concepts[folder]:
                            folders_with_concept.add(folder_names[i])
                    unique_words = set()
                    for fname in folders_with_concept:
                        unique_words |= folder_unique_map[fname]
                    all_concepts_table[concept] = [w.capitalize() for w in unique_words]
            
            # Use the same group data that was used for the Group Presence Matrix
            # This ensures consistency between the two tables
            if self.llm_grouping_var.get():
                for i, row_data in enumerate(group_data):
                    row = uniq_table.add_row().cells
                    
                    # Use the same group name from group_data (which may have suffixes)
                    group_name = unique_group_names[i] if i < len(unique_group_names) else row_data[0]
                    row[0].text = group_name
                    
                    # Find the corresponding concepts and color for this group
                    concepts = None
                    color = None
                    for c, concepts_list in llm_group_tuples:
                        if extract_group_name(concepts_list) == row_data[0]:  # Use original name without suffix
                            concepts = concepts_list
                            color = c
                            break
                    
                    if concepts is None:
                        continue
                    
                    # Concepts column (colored)
                    para = row[1].paragraphs[0]
                    for cidx, concept in enumerate(concepts):
                        rc = para.add_run(concept)
                        if color.startswith('#') and len(color) == 7:
                            r, g, b = tuple(int(color[j:j+2], 16) for j in (1, 3, 5))
                            rc.font.color.rgb = RGBColor(r, g, b)
                        if cidx < len(concepts) - 1:
                            para.add_run(", ")
                    
                    # Get the actual folders where this group appears from group_data
                    present_folders = [folder_names[i] for i, present in enumerate(row_data[1:]) if present]
                    n_present = len(present_folders)
                    n_total = len(folder_names)
                    
                    # Set Unique/Shared status based on actual folder presence
                    if n_present == 1:
                        row[2].text = "Unique"
                    elif n_present == n_total:
                        row[2].text = f"Common in {n_present} / {n_total} authors"
                    else:
                        row[2].text = f"Partial in {n_present} / {n_total} authors"
                    
                    # Set Folders column to show the unique capitalized folder names
                    unique_folder_names = []
                    for folder_name in sorted(present_folders):
                        folder_unique_words = folder_unique_map.get(folder_name, set())
                        if folder_unique_words:
                            short_name = sorted(folder_unique_words)[0].capitalize()
                        else:
                            short_name = folder_name.split()[0].capitalize()
                        unique_folder_names.append(short_name)
                    row[3].text = ", ".join(unique_folder_names)
                    row[4].text = str(len(concepts))
            else:
                for i, row_data in enumerate(group_data):
                    row = uniq_table.add_row().cells
                    
                    # Use the same group name from group_data (which may have suffixes)
                    group_name = unique_group_names[i] if i < len(unique_group_names) else row_data[0]
                    row[0].text = group_name
                    
                    # Find the corresponding concepts and color for this group
                    concepts = None
                    color = None
                    for c, concepts_list in color_to_concepts.items():
                        if extract_group_name(concepts_list) == row_data[0]:  # Use original name without suffix
                            concepts = concepts_list
                            color = c
                            break
                    
                    if concepts is None:
                        continue
                    
                    # Concepts column (colored)
                    para = row[1].paragraphs[0]
                    for cidx, concept in enumerate(concepts):
                        rc = para.add_run(concept)
                        if color.startswith('#') and len(color) == 7:
                            r, g, b = tuple(int(color[j:j+2], 16) for j in (1, 3, 5))
                            rc.font.color.rgb = RGBColor(r, g, b)
                        if cidx < len(concepts) - 1:
                            para.add_run(", ")
                    
                    # Get the actual folders where this group appears from group_data
                    present_folders = [folder_names[i] for i, present in enumerate(row_data[1:]) if present]
                    n_present = len(present_folders)
                    n_total = len(folder_names)
                    
                    # Set Unique/Shared status based on actual folder presence
                    if n_present == 1:
                        row[2].text = "Unique"
                    elif n_present == n_total:
                        row[2].text = f"Common in {n_present} / {n_total} authors"
                    else:
                        row[2].text = f"Partial in {n_present} / {n_total} authors"
                    
                    # Set Folders column to show the unique capitalized folder names
                    unique_folder_names = []
                    for folder_name in sorted(present_folders):
                        folder_unique_words = folder_unique_map.get(folder_name, set())
                        if folder_unique_words:
                            short_name = sorted(folder_unique_words)[0].capitalize()
                        else:
                            short_name = folder_name.split()[0].capitalize()
                        unique_folder_names.append(short_name)
                    row[3].text = ", ".join(unique_folder_names)
                    row[4].text = str(len(concepts))


            doc.add_heading("Aggregated Results", level=1)
            # Table of images
            doc.add_heading("UpSet Plots from Each Folder", level=2)
            img_table = doc.add_table(rows=1, cols=5)
            hdr = img_table.rows[0].cells
            hdr[0].text = "Folder"
            hdr[1].text = "BM25"
            hdr[2].text = "Top-k"
            hdr[3].text = "Top-p"
            hdr[4].text = "Temp"
            for folder_name, imgs in folder_images:
                row = img_table.add_row().cells
                row[0].text = folder_name
                for i, img_path in enumerate(imgs):
                    try:
                        run = row[i+1].paragraphs[0].add_run()
                        run.add_picture(img_path, width=Inches(1.5))
                    except Exception as e:
                        row[i+1].text = "[Image error]"
            doc.add_page_break()
            # Table of all canonical concepts (color-coded, with variants, grouped by color)
            doc.add_heading("All Concepts (Grouped by Color, Color-coded)", level=2)
            # color_to_concepts is already created above
            concept_table = doc.add_table(rows=1, cols=3)
            concept_table.rows[0].cells[0].text = "Color Group"
            concept_table.rows[0].cells[1].text = "Concepts"
            concept_table.rows[0].cells[2].text = "Folder Unique Words"
            for color, concepts in color_to_concepts.items():
                row = concept_table.add_row().cells
                run = row[0].paragraphs[0].add_run(color)
                if color.startswith('#') and len(color) == 7:
                    r, g, b = tuple(int(color[j:j+2], 16) for j in (1, 3, 5))
                    run.font.color.rgb = RGBColor(r, g, b)
                # All concepts in this group in the same color
                para = row[1].paragraphs[0]
                for idx, c in enumerate(concepts):
                    rc = para.add_run(c)
                    color_val = group_colors.get(c, '#FF0000')
                    if c not in group_colors:
                        print(f"[GROUP COLOR WARNING] Concept not in group_colors: {c}")
                        self.root.after(0, lambda c=c: self.aggregate_status_label.config(text=f"[GROUP COLOR WARNING] Concept not in group_colors: {c}", foreground='red'))
                        group_colors[c] = '#FF0000'
                    if color_val.startswith('#') and len(color_val) == 7:
                        rc.font.color.rgb = RGBColor(int(color_val[1:3], 16), int(color_val[3:5], 16), int(color_val[5:7], 16))
                    if idx < len(concepts) - 1:
                        para.add_run(", ")
                # Folder unique words for this group (use same logic as above)
                folders_with_concept = set()
                for c in concepts:
                    for i, folder in enumerate(valid_folders):
                        if c in folder_concepts[folder]:
                            folders_with_concept.add(folder_names[i])
                unique_words = set()
                for fname in folders_with_concept:
                    unique_words |= folder_unique_map[fname]
                row[2].text = ", ".join(sorted(unique_words))
            doc.add_page_break()
            # --- Overlap table: put all concepts in the same color group on the same row ---
            doc.add_heading("Concept Overlap Across Authors (Grouped)", level=2)
            overlap_doc_table = doc.add_table(rows=1, cols=1+len(folder_names))
            hdr = overlap_doc_table.rows[0].cells
            hdr[0].text = "Concepts in Group"
            for i, name in enumerate(folder_names):
                hdr[1+i].text = name
            # Make headings bold
            for cell in overlap_doc_table.rows[0].cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        run.bold = True
            if self.llm_grouping_var.get():
                for idx, (color, concepts) in enumerate(llm_group_tuples):
                    doc_row = overlap_doc_table.add_row().cells
                    para = doc_row[0].paragraphs[0]
                    for cidx, c in enumerate(concepts):
                        rc = para.add_run(c)
                        if color.startswith('#') and len(color) == 7:
                            r, g, b = tuple(int(color[j:j+2], 16) for j in (1, 3, 5))
                            rc.font.color.rgb = RGBColor(r, g, b)
                        if cidx < len(concepts) - 1:
                            para.add_run(", ")
                    concepts_in_color = set(concepts)
                    for j, folder in enumerate(valid_folders):
                        present = any(c in folder_concepts[folder] for c in concepts_in_color)
                        doc_row[1+j].text = "✔" if present else ""
            else:
                for idx, color in enumerate(unique_color_groups):
                    doc_row = overlap_doc_table.add_row().cells
                    para = doc_row[0].paragraphs[0]
                    concepts = color_to_concepts[color]
                    for cidx, c in enumerate(concepts):
                        rc = para.add_run(c)
                        if color.startswith('#') and len(color) == 7:
                            r, g, b = tuple(int(color[j:j+2], 16) for j in (1, 3, 5))
                            rc.font.color.rgb = RGBColor(r, g, b)
                        if cidx < len(concepts) - 1:
                            para.add_run(", ")
                    concepts_in_color = set(concepts)
                    for j, folder in enumerate(valid_folders):
                        present = any(c in folder_concepts[folder] for c in concepts_in_color)
                        doc_row[1+j].text = "✔" if present else ""
            # Final step: Save document
            current_step += 1
            safe_update_progress(current_step, total_steps, total_elapsed, est_total)
            safe_update_status("Saving document...", "blue")
            
            # Save with appropriate filename based on grouping method
            from datetime import datetime
            date_str = datetime.now().strftime("%Y%m%d")
            
            # Determine filename based on grouping method
            if self.llm_grouping_var.get():
                # LLM Grouping selected - use LLM model name
                model_name = self.llm_model_var.get().replace(" ", "_").replace("/", "_")
                out_path = os.path.join(parent, f"aggregated_{model_name}_{date_str}.docx")
            else:
                # Group by words selected - use "words"
                out_path = os.path.join(parent, f"aggregated_words_{date_str}.docx")
            doc.save(out_path)
            
            # Completion - don't increment step, just update progress to 100%
            safe_update_progress(100, total_steps, total_elapsed, total_elapsed)
            safe_update_status(f"Aggregation complete. Saved to {out_path}", "green")
        except Exception as e:
            tb = traceback.format_exc()
            safe_update_status(f"Error generating DOCX: {e}", "red")
            print(tb)

    def _update_aggregate_progress(self, val, elapsed, est_total, current_step=None, total_steps=None, remaining_time=None):
        self.aggregate_progress['value'] = val
        if current_step is not None and total_steps is not None and remaining_time is not None:
            self.aggregate_time_label.config(text=f"Step {current_step}/{total_steps} | Elapsed: {elapsed:.1f}s | Remaining: {remaining_time:.1f}s | Total: {est_total:.1f}s")
        else:
            self.aggregate_time_label.config(text=f"Elapsed: {elapsed:.1f}s, Estimated total: {est_total:.1f}s")
    
    def _add_rag_consistency_analysis(self, doc, valid_folders, folder_concepts, parent_dir):
        """Add RAG consistency analysis tables to the document"""
        try:
            import time
            from docx.shared import Inches
            
            # Check which analyses are enabled
            if not self.analyze_consistency_var.get():
                return
            
            # Get the selected embedding model
            model_display_name = self.embedding_model_var.get()
            model_mapping = {
                "🤗 sentence-transformers/all-MiniLM-L6-v2 (384 dim, fast)": "sentence-transformers/all-MiniLM-L6-v2",
                "🤗 sentence-transformers/all-mpnet-base-v2 (768 dim, high-quality)": "sentence-transformers/all-mpnet-base-v2",
                "🤗 sentence-transformers/all-distilroberta-v1 (768 dim, balanced)": "sentence-transformers/all-distilroberta-v1",
                "🤗 sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384 dim, multilingual)": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "🤗 sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (768 dim, multilingual)": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                "🤗 BAAI/bge-small-en-v1.5 (384 dim, efficient)": "BAAI/bge-small-en-v1.5",
                "🤗 BAAI/bge-base-en-v1.5 (768 dim, excellent)": "BAAI/bge-base-en-v1.5",
                "🤗 BAAI/bge-large-en-v1.5 (1024 dim, powerful)": "BAAI/bge-large-en-v1.5",
                "🤗 intfloat/e5-base-v2 (768 dim, general-purpose)": "intfloat/e5-base-v2",
                "🤗 intfloat/e5-large-v2 (1024 dim, advanced)": "intfloat/e5-large-v2",
                "🟦 Qwen/Qwen3-Embedding-8B (1024 dim, advanced)": "Qwen/Qwen3-Embedding-8B",
                "🟦 BAAI/bge-en-icl (1024 dim, instruction-tuned)": "BAAI/bge-en-icl",
                "🟦 BAAI/bge-multilingual-gemma2 (1024 dim, multilingual)": "BAAI/bge-multilingual-gemma2"
            }
            model_name = model_mapping.get(model_display_name, "sentence-transformers/all-MiniLM-L6-v2")
            
            # Extract parameter data from HTML files (grouped by folder)
            folder_data = self._extract_parameter_data_from_html(valid_folders)
            
            if not folder_data:
                print("No parameter data found for RAG consistency analysis")
                return
            
            # Generate folder-specific analysis
            folder_analysis_results = {}
            analysis_timings = {}
            
            print(f"[RAG ANALYSIS] Analyzing {len(folder_data)} folders individually")
            print(f"[RAG ANALYSIS] Folder data keys: {list(folder_data.keys())}")
            for folder_name, data in folder_data.items():
                print(f"[RAG ANALYSIS] {folder_name}: {len(data)} parameter combinations")
            
            for folder in valid_folders:
                folder_name = os.path.basename(folder)
                print(f"[RAG ANALYSIS] Processing folder: {folder_name}")
                
                if folder_name not in folder_data:
                    print(f"[RAG ANALYSIS] No data found for folder: {folder_name}")
                    continue
                
                # Create analyzer for this folder
                folder_analyzer = RAGConsistencyAnalyzer(model_name)
                
                # Add data for this folder only
                print(f"[RAG ANALYSIS] Adding {len(folder_data[folder_name])} parameter combinations to analyzer")
                for i, param_combo in enumerate(folder_data[folder_name]):
                    concepts_text = " ".join(param_combo['concepts'])
                    print(f"[RAG ANALYSIS] Adding result {i+1}: T={param_combo['params']['temp']}, P={param_combo['params']['topp']}, K={param_combo['params']['topk']}, B={param_combo['params']['bm25']}, Concepts={len(param_combo['concepts'])}")
                    folder_analyzer.add_result(
                        param_combo['params']['temp'],
                        param_combo['params']['topp'],
                        param_combo['params']['topk'],
                        param_combo['params']['bm25'],
                        concepts_text
                    )
                print(f"[RAG ANALYSIS] Analyzer now has {len(folder_analyzer.data)} data points")
                
                if folder_analyzer.data:
                    # Time the full analysis process including generate_stability_report AND visualizations
                    analysis_start_time = time.time()
                    folder_timings = {}
                    
                    # Generate full report (this does the actual work)
                    folder_results = folder_analyzer.generate_stability_report(
                        run_within_param=self.within_param_var.get(),
                        run_cross_param=self.cross_param_var.get(),
                        run_sensitivity=self.sensitivity_var.get()
                    )
                    
                    # Generate CSV files for enabled analyses
                    csv_files = {}
                    if self.within_param_var.get() and folder_results:
                        csv_files['within_param'] = folder_analyzer.generate_analysis_csv(
                            'within_param', folder_results, folder_name, parent_dir)
                    if self.cross_param_var.get() and folder_results:
                        csv_files['cross_param'] = folder_analyzer.generate_analysis_csv(
                            'cross_param', folder_results, folder_name, parent_dir)
                    if self.sensitivity_var.get() and folder_results:
                        csv_files['sensitivity'] = folder_analyzer.generate_analysis_csv(
                            'sensitivity', folder_results, folder_name, parent_dir)
                    
                    # Calculate total analysis time (excluding cross-author analysis)
                    analysis_end_time = time.time()
                    total_analysis_time = analysis_end_time - analysis_start_time
                    
                    # Distribute timing across enabled analyses
                    enabled_analyses = []
                    if self.within_param_var.get():
                        enabled_analyses.append('within_param')
                    if self.cross_param_var.get():
                        enabled_analyses.append('cross_param')
                    if self.sensitivity_var.get():
                        enabled_analyses.append('sensitivity')
                    
                    # Distribute time evenly across enabled analyses
                    if enabled_analyses:
                        time_per_analysis = total_analysis_time / len(enabled_analyses)
                        for analysis in enabled_analyses:
                            folder_timings[analysis] = time_per_analysis
                    
                    folder_timings['total'] = total_analysis_time
                    
                    folder_analysis_results[folder_name] = folder_results
                    analysis_timings[folder_name] = folder_timings
                    
                    # Debug: Print parameter values found for this folder
                    print(f"[RAG ANALYSIS] {folder_name} - Found {len(folder_analyzer.data)} data points in {total_analysis_time:.2f}s total")
                    
                    # Check if individual_parameters exists before accessing it
                    if 'individual_parameters' in folder_results:
                        for param in ['temperature', 'top_p', 'top_k', 'bm25_weight']:
                            if param in folder_results['individual_parameters']:
                                result = folder_results['individual_parameters'][param]
                                print(f"[RAG ANALYSIS] {folder_name} - {param} values: {result['parameter_values']}")
                    else:
                        print(f"[RAG ANALYSIS] {folder_name} - No individual_parameters found in results")
                        print(f"[RAG ANALYSIS] {folder_name} - Available keys: {list(folder_results.keys())}")
                else:
                    print(f"[RAG ANALYSIS] No data for folder: {folder_name}")
            
            if not folder_analysis_results:
                print("No folder analysis results available")
                return
            
            # Generate cross-author analysis (checkbox D) - done once for all folders
            if self.semantic_viz_var.get():
                print(f"[CROSS-AUTHOR] Generating cross-author analysis for all folders...")
                cross_author_start_time = time.time()
                try:
                    # Create a single analyzer for cross-author analysis
                    cross_analyzer = RAGConsistencyAnalyzer(model_name)
                    
                    # Generate cross-author semantic similarity analysis
                    # Get t-SNE configuration from GUI
                    tsne_font_size = getattr(self, 'tsne_font_size_var', None)
                    tsne_font_size = tsne_font_size.get() if tsne_font_size else 5
                    tsne_n_components = getattr(self, 'tsne_n_components_var', None)
                    tsne_n_components = tsne_n_components.get() if tsne_n_components else 2
                    tsne_color_palette = getattr(self, 'tsne_color_palette_var', None)
                    tsne_color_palette = tsne_color_palette.get() if tsne_color_palette else 'Set3'
                    tsne_proximity = getattr(self, 'tsne_proximity_var', None)
                    tsne_proximity = tsne_proximity.get() if tsne_proximity else None
                    if tsne_proximity is not None and tsne_proximity <= 0:
                        tsne_proximity = None
                    tsne_show_labels = getattr(self, 'tsne_show_labels_var', None)
                    tsne_show_labels = tsne_show_labels.get() if tsne_show_labels else True
                    
                    # Generate cross-author semantic similarity analysis with t-SNE parameters
                    cross_author_results = cross_analyzer.generate_cross_author_analysis(
                        parent_dir, "all_folders", valid_folders, folder_concepts,
                        tsne_font_size=tsne_font_size,
                        tsne_n_components=tsne_n_components,
                        tsne_color_palette=tsne_color_palette,
                        tsne_proximity_threshold=tsne_proximity,
                        tsne_show_labels=tsne_show_labels
                    )
                    
                    # Store visualization paths for later inclusion in DOC
                    if not hasattr(self, 'cross_author_visualizations'):
                        self.cross_author_visualizations = {}
                    self.cross_author_visualizations['all_folders'] = cross_author_results
                    
                    cross_author_end_time = time.time()
                    cross_author_time = cross_author_end_time - cross_author_start_time
                    print(f"[CROSS-AUTHOR] Cross-author analysis completed in {cross_author_time:.2f}s")
                    
                    # Store timing in analysis_timings for display
                    if not hasattr(self, 'analysis_timings'):
                        self.analysis_timings = {}
                    if 'all_folders' not in self.analysis_timings:
                        self.analysis_timings['all_folders'] = {}
                    self.analysis_timings['all_folders']['semantic_viz'] = cross_author_time
                    
                except Exception as cross_error:
                    print(f"[CROSS-AUTHOR ERROR] Failed to generate cross-author analysis: {cross_error}")
                    import traceback
                    traceback.print_exc()
            
            # Add analysis tables to document with timings
            self._add_analysis_tables_to_doc(doc, folder_analysis_results, valid_folders, analysis_timings)
            
            
            # Add semantic similarity visualizations to document (only if Option D is checked)
            if self.semantic_viz_var.get():
                self._add_semantic_visualizations_to_doc(doc)
            
        except Exception as e:
            print(f"Error in RAG consistency analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_parameter_data_from_html(self, valid_folders):
        """Extract parameter data from HTML files in the folders"""
        folder_data = {}
        
        try:
            import bs4
            
            for folder in valid_folders:
                folder_name = os.path.basename(folder)
                html_path = os.path.join(folder, "compare.htm")
                
                if not os.path.exists(html_path):
                    print(f"HTML file not found: {html_path}")
                    continue
                
                print(f"Processing HTML file: {html_path}")
                
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                soup = bs4.BeautifulSoup(html_content, 'html.parser')
                table = soup.find('table')
                
                if not table:
                    print(f"No table found in {html_path}")
                    continue
                
                # Initialize folder data
                folder_data[folder_name] = []
                
                # Parse HTML table rows
                rows = table.find_all('tr')
                print(f"Found {len(rows)-1} data rows in {html_path}")
                
                for idx, row in enumerate(rows[1:], 1):  # Skip header row
                    cells = row.find_all('td')
                    if len(cells) < 5:
                        continue
                        
                    temp = float(cells[0].text.strip())
                    topp = float(cells[1].text.strip())
                    topk = int(cells[2].text.strip())
                    bm25 = float(cells[3].text.strip())
                    
                    # Parse concepts from the last cell
                    concepts_cell = cells[4]
                    concepts = []
                    concept_boxes = concepts_cell.find_all('span', class_='concept-box')
                    
                    for box in concept_boxes:
                        concept_name = box.text.strip()
                        concepts.append(concept_name)
                    
                    print(f"Row {idx}: T={temp}, P={topp}, K={topk}, B={bm25}, Concepts={len(concepts)}")
                    
                    # Store this parameter combination for this folder
                    folder_data[folder_name].append({
                        'params': {'temp': temp, 'topp': topp, 'topk': topk, 'bm25': bm25},
                        'concepts': set(concepts),
                        'total_concepts': len(concepts)
                    })
                
                # Debug: Print unique parameter values found for this folder
                if folder_data[folder_name]:
                    temps = sorted(set([p['params']['temp'] for p in folder_data[folder_name]]))
                    topps = sorted(set([p['params']['topp'] for p in folder_data[folder_name]]))
                    topks = sorted(set([p['params']['topk'] for p in folder_data[folder_name]]))
                    bm25s = sorted(set([p['params']['bm25'] for p in folder_data[folder_name]]))
                    print(f"[HTML EXTRACTION] {folder_name} - Temp: {temps}, TopP: {topps}, TopK: {topks}, BM25: {bm25s}")
            
            print(f"Extracted data for {len(folder_data)} folders")
            return folder_data
            
        except Exception as e:
            print(f"Error extracting parameter data from HTML: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _add_analysis_tables_to_doc(self, doc, folder_analysis_results, valid_folders, analysis_timings=None):
        """Add the three analysis tables to the document"""
        try:
            import time
            from docx.shared import Inches
            from docx.enum.text import WD_ALIGN_PARAGRAPH
            
            # Add header explaining what the analysis does
            doc.add_heading("RAG Parameter Consistency Analysis", level=2)
            doc.add_paragraph(
                "Evaluates concept extraction consistency across parameter settings using semantic similarity and exact matching. "
                "Higher values = more consistent extraction."
            )
            
            # A. Within-Parameter Analysis
            if self.within_param_var.get():
                start_time = time.time()
                doc.add_heading("A. Within-Parameter Analysis", level=3)
                para = doc.add_paragraph()
                para.add_run("Consistency when varying each parameter. Higher values = less impact on concept variation. ").font.size = Inches(0.12)
                para.add_run("Example: Temperature across 0.1, 0.33, 0.78, 1.0.").font.size = Inches(0.12)
                
                # Create within-parameter table
                within_table = doc.add_table(rows=1, cols=len(valid_folders) + 1)
                within_table.style = 'Table Grid'
                
                # Set column widths
                for i, col in enumerate(within_table.columns):
                    if i == 0:
                        col.width = Inches(1.5)  # Parameter column
                    else:
                        col.width = Inches(1.0)  # Folder columns
                
                # Header row
                header_cells = within_table.rows[0].cells
                header_cells[0].text = "Parameter"
                for i, folder in enumerate(valid_folders):
                    header_cells[i + 1].text = os.path.basename(folder)
                
                # Make header bold and set font size
                for cell in within_table.rows[0].cells:
                    for para in cell.paragraphs:
                        for run in para.runs:
                            run.bold = True
                            run.font.size = Inches(0.10)  # Smaller font size
                
                # Add data rows
                parameters = ['temperature', 'top_p', 'top_k', 'bm25_weight']
                for param in parameters:
                    row = within_table.add_row()
                    row.cells[0].text = param.replace('_', ' ').title()
                    
                    # Add data for each folder
                    for i, folder in enumerate(valid_folders):
                        folder_name = os.path.basename(folder)
                        if (folder_name in folder_analysis_results and 
                            'individual_parameters' in folder_analysis_results[folder_name] and
                            param in folder_analysis_results[folder_name]['individual_parameters']):
                            result = folder_analysis_results[folder_name]['individual_parameters'][param]
                            semantic_mean = result.get('overall_semantic_mean', 0.0)
                            exact_mean = result.get('overall_exact_mean', 0.0)
                            total_comparisons = result.get('total_comparisons', 0)
                            row.cells[i + 1].text = f"S: {semantic_mean:.3f}\nE: {exact_mean:.3f}\nC: {total_comparisons}"
                        else:
                            row.cells[i + 1].text = "N/A"
                    
                    # Set font size for all cells in this row
                    for cell in row.cells:
                        for para in cell.paragraphs:
                            for run in para.runs:
                                run.font.size = Inches(0.10)  # Smaller font size
                
                # Add timing information
                timing_para = doc.add_paragraph()
                timing_para.add_run("⏱️ Within-Parameter Analysis completed in ").font.size = Inches(0.12)
                # Get timing from analysis_timings if available
                if analysis_timings and any('within_param' in timings for timings in analysis_timings.values()):
                    times = [timings.get('within_param', 0) for timings in analysis_timings.values() if 'within_param' in timings]
                    if times:
                        total_time_minutes = sum(times) / 60.0  # Total time, not average
                        timing_para.add_run(f"{total_time_minutes:.2f} minutes").font.size = Inches(0.12)
                    else:
                        timing_para.add_run("X.XX minutes").font.size = Inches(0.12)
                else:
                    timing_para.add_run("X.XX minutes").font.size = Inches(0.12)
                timing_para.add_run(".").font.size = Inches(0.12)
            
            # B. Cross-Parameter Analysis
            if self.cross_param_var.get():
                start_time = time.time()
                doc.add_heading("B. Cross-Parameter Analysis", level=3)
                para = doc.add_paragraph()
                para.add_run("Compares ALL parameter combinations pairwise for overall system stability. ").font.size = Inches(0.12)
                para.add_run("Shows overall consistency across all variations.").font.size = Inches(0.12)
                
                # Create cross-parameter table
                cross_table = doc.add_table(rows=1, cols=len(valid_folders) + 1)
                cross_table.style = 'Table Grid'
                
                # Set column widths
                for i, col in enumerate(cross_table.columns):
                    if i == 0:
                        col.width = Inches(1.5)  # Metric column
                    else:
                        col.width = Inches(1.0)  # Folder columns
                
                # Header row
                header_cells = cross_table.rows[0].cells
                header_cells[0].text = "Metric"
                for i, folder in enumerate(valid_folders):
                    header_cells[i + 1].text = os.path.basename(folder)
                
                # Make header bold and set font size
                for cell in cross_table.rows[0].cells:
                    for para in cell.paragraphs:
                        for run in para.runs:
                            run.bold = True
                            run.font.size = Inches(0.10)  # Smaller font size
                
                # Add data rows
                metrics = [
                    ('Semantic Similarity', 'overall_semantic_mean', 'overall_semantic_std'),
                    ('Exact Matching', 'overall_exact_mean', 'overall_exact_std'),
                    ('Total Comparisons', 'total_comparisons', None)
                ]
                
                for metric_name, mean_key, std_key in metrics:
                    row = cross_table.add_row()
                    row.cells[0].text = metric_name
                    
                    # Add data for each folder
                    for i, folder in enumerate(valid_folders):
                        folder_name = os.path.basename(folder)
                        if folder_name in folder_analysis_results and 'cross_parameter' in folder_analysis_results[folder_name]:
                            result = folder_analysis_results[folder_name]['cross_parameter']
                            mean_val = result.get(mean_key, 0.0)
                            if std_key and std_key in result:
                                std_val = result.get(std_key, 0.0)
                                row.cells[i + 1].text = f"{mean_val:.3f} ± {std_val:.3f}"
                            else:
                                row.cells[i + 1].text = f"{mean_val:.0f}" if mean_key == 'total_comparisons' else f"{mean_val:.3f}"
                        else:
                            row.cells[i + 1].text = "N/A"
                    
                    # Set font size for all cells in this row
                    for cell in row.cells:
                        for para in cell.paragraphs:
                            for run in para.runs:
                                run.font.size = Inches(0.10)  # Smaller font size
                
                # Add timing information
                timing_para = doc.add_paragraph()
                timing_para.add_run("⏱️ Cross-Parameter Analysis completed in ").font.size = Inches(0.12)
                # Get timing from analysis_timings if available
                if analysis_timings and any('cross_param' in timings for timings in analysis_timings.values()):
                    times = [timings.get('cross_param', 0) for timings in analysis_timings.values() if 'cross_param' in timings]
                    if times:
                        total_time_minutes = sum(times) / 60.0  # Total time, not average
                        timing_para.add_run(f"{total_time_minutes:.2f} minutes").font.size = Inches(0.12)
                    else:
                        timing_para.add_run("X.XX minutes").font.size = Inches(0.12)
                else:
                    timing_para.add_run("X.XX minutes").font.size = Inches(0.12)
                timing_para.add_run(".").font.size = Inches(0.12)
            
            # C. Parameter Sensitivity Ranking
            if self.sensitivity_var.get():
                start_time = time.time()
                doc.add_heading("C. Parameter Sensitivity Ranking", level=3)
                doc.add_paragraph(
                    "Shows which parameters cause the most concept variation. "
                    "Lower values indicate higher sensitivity (more variation when parameter changes)."
                )
                
                # Create sensitivity table
                sensitivity_table = doc.add_table(rows=1, cols=len(valid_folders) + 1)
                sensitivity_table.style = 'Table Grid'
                
                # Set column widths
                for i, col in enumerate(sensitivity_table.columns):
                    if i == 0:
                        col.width = Inches(1.5)  # Parameter column
                    else:
                        col.width = Inches(1.0)  # Folder columns
                
                # Header row
                header_cells = sensitivity_table.rows[0].cells
                header_cells[0].text = "Parameter"
                for i, folder in enumerate(valid_folders):
                    header_cells[i + 1].text = os.path.basename(folder)
                
                # Make header bold
                for cell in sensitivity_table.rows[0].cells:
                    for para in cell.paragraphs:
                        for run in para.runs:
                            run.bold = True
                
                # Calculate sensitivity rankings (use first folder for ranking order)
                first_folder = os.path.basename(valid_folders[0]) if valid_folders else None
                if (first_folder and first_folder in folder_analysis_results and 
                    'individual_parameters' in folder_analysis_results[first_folder]):
                    first_folder_results = folder_analysis_results[first_folder]['individual_parameters']
                    # Sort parameters by sensitivity (lower consistency = higher sensitivity)
                    param_sensitivities = []
                    for param in ['temperature', 'top_p', 'top_k', 'bm25_weight']:
                        if param in first_folder_results:
                            result = first_folder_results[param]
                            consistency = result.get('overall_semantic_mean', 0.0)
                            param_sensitivities.append((param, consistency))
                    
                    # Sort by consistency (ascending - most sensitive first)
                    param_sensitivities.sort(key=lambda x: x[1])
                    sorted_params = [p[0] for p in param_sensitivities]
                else:
                    sorted_params = ['temperature', 'top_p', 'top_k', 'bm25_weight']
                
                # Add data rows
                for param in sorted_params:
                    row = sensitivity_table.add_row()
                    row.cells[0].text = param.replace('_', ' ').title()
                    
                    # Add data for each folder
                    for i, folder in enumerate(valid_folders):
                        folder_name = os.path.basename(folder)
                        if (folder_name in folder_analysis_results and 
                            'individual_parameters' in folder_analysis_results[folder_name] and
                            param in folder_analysis_results[folder_name]['individual_parameters']):
                            result = folder_analysis_results[folder_name]['individual_parameters'][param]
                            semantic_mean = result.get('overall_semantic_mean', 0.0)
                            exact_mean = result.get('overall_exact_mean', 0.0)
                            total_comparisons = result.get('total_comparisons', 0)
                            row.cells[i + 1].text = f"S: {semantic_mean:.3f}\nE: {exact_mean:.3f}\nC: {total_comparisons}"
                        else:
                            row.cells[i + 1].text = "N/A"
                
                # Add timing information
                timing_para = doc.add_paragraph()
                timing_para.add_run("⏱️ Parameter Sensitivity Ranking completed in ").font.size = Inches(0.12)
                # Get timing from analysis_timings if available
                if analysis_timings and any('sensitivity' in timings for timings in analysis_timings.values()):
                    times = [timings.get('sensitivity', 0) for timings in analysis_timings.values() if 'sensitivity' in timings]
                    if times:
                        total_time_minutes = sum(times) / 60.0  # Total time, not average
                        timing_para.add_run(f"{total_time_minutes:.2f} minutes").font.size = Inches(0.12)
                    else:
                        timing_para.add_run("X.XX minutes").font.size = Inches(0.12)
                else:
                    timing_para.add_run("X.XX minutes").font.size = Inches(0.12)
                timing_para.add_run(".").font.size = Inches(0.12)
            
            # D. Semantic Similarity Across Authors / Folders
            if self.semantic_viz_var.get():
                start_time = time.time()
                doc.add_heading("D. Semantic Similarity Across Authors / Folders", level=3)
                doc.add_paragraph(
                    "This analysis examines semantic similarity between different authors/folders using cross-author analysis. "
                    "It generates heatmaps, 2D embeddings, and network graphs to show conceptual relationships across authors, "
                    "helping identify shared conceptual spaces and author-specific concept patterns."
                )
                
                # Add semantic similarity matrix table if available
                if (hasattr(self, 'cross_author_visualizations') and 
                    self.cross_author_visualizations and 
                    'all_folders' in self.cross_author_visualizations):
                    viz_data = self.cross_author_visualizations['all_folders']
                    if viz_data and isinstance(viz_data, dict) and 'between_author_sim' in viz_data:
                        between_author_sim = viz_data['between_author_sim']
                        folder_names = list(valid_folders)
                        folder_names = [os.path.basename(f) for f in folder_names]
                        
                        # Create similarity matrix table
                        doc.add_heading("Semantic Similarity Matrix", level=4)
                        doc.add_paragraph(
                            "This table shows the semantic similarity values between all author/folder pairs. "
                            "Values range from 0 (no similarity) to 1 (identical conceptual space)."
                        )
                        
                        # Create table with similarity values
                        sim_table = doc.add_table(rows=len(folder_names) + 1, cols=len(folder_names) + 1)
                        sim_table.style = 'Table Grid'
                        
                        # Set column widths
                        for i, col in enumerate(sim_table.columns):
                            if i == 0:
                                col.width = Inches(1.5)  # Author name column
                            else:
                                col.width = Inches(1.0)  # Similarity value columns
                        
                        # Header row
                        header_cells = sim_table.rows[0].cells
                        header_cells[0].text = "Author/Folder"
                        for i, folder in enumerate(folder_names):
                            header_cells[i + 1].text = folder
                        
                        # Make header bold
                        for cell in sim_table.rows[0].cells:
                            for para in cell.paragraphs:
                                for run in para.runs:
                                    run.bold = True
                        
                        # Data rows
                        for i, folder1 in enumerate(folder_names):
                            row = sim_table.rows[i + 1]
                            row.cells[0].text = folder1
                            for j, folder2 in enumerate(folder_names):
                                if i < len(between_author_sim) and j < len(between_author_sim[i]):
                                    sim_value = between_author_sim[i][j]
                                    row.cells[j + 1].text = f"{sim_value:.3f}"
                                else:
                                    row.cells[j + 1].text = "N/A"
            
            
        except Exception as e:
            print(f"Error adding analysis tables to document: {e}")
            import traceback
            traceback.print_exc()

    def _add_semantic_visualizations_to_doc(self, doc):
        """Add semantic similarity visualizations to the document"""
        try:
            from docx.shared import Inches
            from docx.enum.text import WD_ALIGN_PARAGRAPH
            
            # Check if we have any visualizations to add
            has_visualizations = False
            
            # Check for cross-author visualizations (checkbox D)
            if hasattr(self, 'cross_author_visualizations') and self.cross_author_visualizations:
                # Check if there's actual data
                for folder_name, viz_data in self.cross_author_visualizations.items():
                    if viz_data and isinstance(viz_data, dict):
                        has_visualizations = True
                        break
                
                if has_visualizations:
                    self._add_cross_author_visualizations_to_doc(doc)
                else:
                    print("[SEMANTIC VIZ] Cross-author visualizations dictionary exists but contains no valid data")
                    doc.add_paragraph(
                        "Note: Cross-author visualizations were requested but could not be generated. "
                        "This may be due to insufficient data or an error during analysis."
                    )
            
            # Check for individual folder visualizations (if any)
            if hasattr(self, 'semantic_visualizations') and self.semantic_visualizations:
                has_individual = False
                for folder_name, viz_paths in self.semantic_visualizations.items():
                    if viz_paths and (viz_paths.get('heatmap') or viz_paths.get('tsne')):
                        has_individual = True
                        break
                
                if has_individual:
                    has_visualizations = True
                    self._add_individual_folder_visualizations_to_doc(doc)
            
            if not has_visualizations:
                print("[SEMANTIC VIZ] No visualizations to add to document")
                doc.add_paragraph(
                    "Note: No visualizations were generated. This may be due to insufficient data or an error during analysis."
                )
                return
            
        except Exception as e:
            print(f"[SEMANTIC VIZ ERROR] Failed to add visualizations to document: {e}")
            import traceback
            traceback.print_exc()

    def _add_cross_author_visualizations_to_doc(self, doc):
        """Add cross-author visualizations to the document"""
        try:
            from docx.shared import Inches
            from docx.enum.text import WD_ALIGN_PARAGRAPH
            
            # Add cross-author analysis header
            doc.add_heading("Cross-Author Semantic Analysis", level=2)
            doc.add_paragraph(
                "This section shows semantic similarity analysis across all authors/folders. "
                "These visualizations help identify shared conceptual spaces and author-specific patterns."
            )
            
            for folder_name, viz_data in self.cross_author_visualizations.items():
                if not viz_data or not isinstance(viz_data, dict):
                    continue
                
                # Create a 2-column table for t-SNE and Network visualizations
                table = doc.add_table(rows=2, cols=2)
                table.style = 'Table Grid'
                table.autofit = False
                
                # Calculate available table width (page width - margins) - same as other tables
                section = doc.sections[0]
                table_width = int(section.page_width - section.left_margin - section.right_margin)
                
                # Set column widths to use full page width (50% each)
                col_width = int(table_width // 2)  # Each column takes 50% of available width
                for col in table.columns:
                    col.width = col_width
                
                # Row 1: Headers and descriptions
                headers = [
                    "Cross-Author Concept Clustering (t-SNE)", 
                    "Semantic Concept Network"
                ]
                
                descriptions = [
                    "2D scatter plot showing concept clustering by author. Points closer together represent more semantically similar concepts. Different colors represent different authors. This is a dimensionality reduction of the high-dimensional concept embeddings into 2D space for visualization.",
                    "Network graph showing relationships between semantically similar concepts. Nodes represent concepts, edges represent high similarity (>0.8). Node positions are determined by a spring layout algorithm that places connected nodes closer together. Different colors represent different authors."
                ]
                
                for i, (header, desc) in enumerate(zip(headers, descriptions)):
                    cell = table.cell(0, i)
                    cell.paragraphs[0].text = header
                    cell.paragraphs[0].runs[0].bold = True
                    cell.paragraphs[0].runs[0].font.size = Inches(0.12)
                    
                    # Add description
                    desc_para = cell.add_paragraph()
                    desc_para.text = desc
                    desc_para.runs[0].font.size = Inches(0.10)
                
                # Row 2: Images
                images = [
                    ('tsne', 't-SNE'),
                    ('network', 'network')
                ]
                
                # Add alternative visualization if available
                if viz_data.get('alternative_viz') and os.path.exists(viz_data['alternative_viz']):
                    # Add after the table
                    doc.add_heading("Alternative Clustering Visualization (Less Crowded)", level=3)
                    doc.add_paragraph(
                        "This visualization shows concept clusters using cluster centers with pie charts showing author composition. "
                        "Pie charts show the proportion of each author in the cluster, and individual concepts are shown as colored points."
                    )
                    try:
                        doc.add_picture(viz_data['alternative_viz'], width=Inches(6))
                        last_paragraph = doc.paragraphs[-1]
                        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    except Exception as img_error:
                        doc.add_paragraph(f"[Error loading alternative visualization: {img_error}]")
                
                # Add varying radius visualization if available
                if viz_data.get('varying_radius_viz') and os.path.exists(viz_data['varying_radius_viz']):
                    doc.add_heading("Varying Radius Clustering Visualization", level=3)
                    doc.add_paragraph(
                        "This visualization shows concept clusters as circles with varying radius based on the number of cluster members. "
                        "Larger circles represent clusters with more concepts. Circle colors represent the dominant author or a weighted "
                        "average of author colors for mixed clusters. Circles are automatically sized to prevent overlap."
                    )
                    try:
                        doc.add_picture(viz_data['varying_radius_viz'], width=Inches(6))
                        last_paragraph = doc.paragraphs[-1]
                        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    except Exception as img_error:
                        doc.add_paragraph(f"[Error loading varying radius visualization: {img_error}]")
                
                for i, (img_key, img_name) in enumerate(images):
                    cell = table.cell(1, i)
                    
                    if viz_data.get(img_key) and os.path.exists(viz_data[img_key]):
                        try:
                            # Add image with full column width (slightly smaller to fit within cell padding)
                            # Convert column width from EMU to inches for image sizing
                            img_width_inches = col_width / 914400  # 914400 EMU = 1 inch
                            cell.paragraphs[0].add_run().add_picture(viz_data[img_key], width=Inches(img_width_inches * 0.9))
                            # Center the image
                            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                        except Exception as img_error:
                            cell.paragraphs[0].text = f"[Error loading {img_name} image: {img_error}]"
                    else:
                        cell.paragraphs[0].text = f"[{img_name} image not available]"
                
                # Add Between-Author Similarity Heatmap below the table
                if viz_data.get('heatmap') and os.path.exists(viz_data['heatmap']):
                    doc.add_heading("Between-Author Similarity Heatmap", level=3)
                    doc.add_paragraph(
                        "This heatmap shows semantic similarity between different authors. "
                        "Darker colors indicate higher semantic overlap between authors. "
                        "Values range from 0 (no similarity) to 1 (identical conceptual space)."
                    )
                    
                    try:
                        doc.add_picture(viz_data['heatmap'], width=Inches(5))
                        last_paragraph = doc.paragraphs[-1]
                        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    except Exception as img_error:
                        doc.add_paragraph(f"[Error loading heatmap image: {img_error}]")
                
                # Add CSV information below the table
                if viz_data.get('csv') and os.path.exists(viz_data['csv']):
                    doc.add_heading("Cross-Author Analysis Data", level=3)
                    doc.add_paragraph(
                        f"Detailed analysis results are available in: {os.path.basename(viz_data['csv'])}"
                    )
                    doc.add_paragraph(
                        "This file contains:"
                    )
                    doc.add_paragraph("• cross_author_analysis.csv - Between-author similarity matrix", style='List Bullet')
                    doc.add_paragraph("• cross_author_concepts_summary.csv - Concept summaries for each author", style='List Bullet')
                
                # Add spacing
                doc.add_paragraph()
                
                # Add timing information at the end of section D
                if hasattr(self, 'analysis_timings') and self.analysis_timings:
                    # Add Cross-Author Analysis timing
                    if any('semantic_viz' in timings for timings in self.analysis_timings.values()):
                        timing_para = doc.add_paragraph()
                        timing_para.add_run("⏱️ Cross-Author Analysis completed in ").font.size = Inches(0.12)
                        times = [timings.get('semantic_viz', 0) for timings in self.analysis_timings.values() if 'semantic_viz' in timings]
                        if times:
                            total_time_minutes = sum(times) / 60.0
                            timing_para.add_run(f"{total_time_minutes:.2f} minutes").font.size = Inches(0.12)
                        else:
                            timing_para.add_run("X.XX minutes").font.size = Inches(0.12)
                        timing_para.add_run(".").font.size = Inches(0.12)
                    
                    # Add Total RAG Consistency Analysis timing
                    if hasattr(self, 'analysis_timings') and self.analysis_timings:
                        total_times = []
                        for timings in self.analysis_timings.values():
                            if 'total' in timings:
                                total_times.append(timings['total'])
                            else:
                                # Fallback: sum all individual timings
                                total_times.append(sum(timings.values()))
                        
                        if total_times:
                            total_time_minutes = sum(total_times) / 60.0
                            timing_para = doc.add_paragraph()
                            timing_para.add_run("⏱️ Total RAG Consistency Analysis completed in ").font.size = Inches(0.12)
                            timing_para.add_run(f"{total_time_minutes:.2f} minutes").font.size = Inches(0.12)
                            timing_para.add_run(".").font.size = Inches(0.12)
                            
                            # Add per-folder breakdown
                            timing_para2 = doc.add_paragraph()
                            timing_para2.add_run("Per-folder breakdown: ").font.size = Inches(0.12)
                            folder_times = []
                            for folder_name, timings in self.analysis_timings.items():
                                if 'total' in timings:
                                    folder_time_min = timings['total'] / 60.0
                                    folder_times.append(f"{folder_name}: {folder_time_min:.2f}min")
                            timing_para2.add_run("; ".join(folder_times)).font.size = Inches(0.12)
                    
            
            print("[CROSS-AUTHOR VIZ] Successfully added cross-author visualizations to document")
            
        except Exception as e:
            print(f"[CROSS-AUTHOR VIZ ERROR] Failed to add cross-author visualizations: {e}")
            import traceback
            traceback.print_exc()

    def _add_individual_folder_visualizations_to_doc(self, doc):
        """Add individual folder visualizations to the document"""
        try:
            from docx.shared import Inches
            from docx.enum.text import WD_ALIGN_PARAGRAPH
            
            # Add header for individual folder visualizations
            doc.add_heading("Individual Folder Visualizations", level=2)
            doc.add_paragraph(
                "Visual representations of concept similarity within individual folders. "
                "These visualizations help understand how concepts cluster within each author's work."
            )
            
            # Add visualizations for each folder
            for folder_name, viz_paths in self.semantic_visualizations.items():
                if not viz_paths['heatmap'] and not viz_paths['tsne']:
                    continue
                
                # Add folder header
                doc.add_heading(f"Folder: {folder_name}", level=3)
                
                # Add heatmap if available
                if viz_paths['heatmap'] and os.path.exists(viz_paths['heatmap']):
                    doc.add_heading("Semantic Similarity Matrix Heatmap", level=4)
                    doc.add_paragraph(
                        "This heatmap shows the cosine similarity between all concept pairs. "
                        "Darker colors indicate higher semantic similarity. Values range from 0 (no similarity) to 1 (identical meaning)."
                    )
                    
                    # Add the image
                    try:
                        doc.add_picture(viz_paths['heatmap'], width=Inches(6))
                        # Center the image
                        last_paragraph = doc.paragraphs[-1]
                        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    except Exception as img_error:
                        doc.add_paragraph(f"[Error loading heatmap image: {img_error}]")
                
                # Add t-SNE plot if available
                if viz_paths['tsne'] and os.path.exists(viz_paths['tsne']):
                    doc.add_heading("t-SNE Concept Clustering", level=4)
                    doc.add_paragraph(
                        "This 2D scatter plot shows how concepts cluster based on semantic similarity. "
                        "Points closer together represent more semantically similar concepts. "
                        "The color intensity indicates concept frequency across parameter variations."
                    )
                    
                    # Add the image
                    try:
                        doc.add_picture(viz_paths['tsne'], width=Inches(6))
                        # Center the image
                        last_paragraph = doc.paragraphs[-1]
                        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    except Exception as img_error:
                        doc.add_paragraph(f"[Error loading t-SNE image: {img_error}]")
                
                # Add spacing between folders
                doc.add_paragraph()
            
            print("[INDIVIDUAL VIZ] Successfully added individual folder visualizations to document")
            
        except Exception as e:
            print(f"[INDIVIDUAL VIZ ERROR] Failed to add individual folder visualizations: {e}")
            import traceback
            traceback.print_exc()

    def _find_parameter_csv_files(self, valid_folders):
        """Find CSV files containing parameter names in each folder"""
        parameter_csv_files = {}
        
        # Define parameter patterns to look for in filenames
        parameter_patterns = {
            'TEMP': ['temp', 'temperature'],
            'TOPP': ['topp', 'top_p'],
            'TOPK': ['topk', 'top_k'],
            'BM25': ['bm25', 'bm25_weight']
        }
        
        for folder in valid_folders:
            folder_name = os.path.basename(folder)
            parameter_csv_files[folder_name] = {}
            
            try:
                # Look for CSV files in the folder
                csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
                
                for csv_file in csv_files:
                    csv_path = os.path.join(folder, csv_file)
                    csv_lower = csv_file.lower()
                    
                    # Check which parameter this CSV file corresponds to
                    for param_key, patterns in parameter_patterns.items():
                        if any(pattern in csv_lower for pattern in patterns):
                            parameter_csv_files[folder_name][param_key] = csv_path
                            print(f"[QUOTES] Found {param_key} CSV: {csv_file} in {folder_name}")
                            break
                            
            except Exception as e:
                print(f"[QUOTES ERROR] Error scanning folder {folder_name}: {e}")
        
        return parameter_csv_files

    def _parse_csv_for_quotes(self, csv_path, concept_name):
        """Parse CSV file to extract quotes for a specific concept"""
        quotes = []
        
        try:
            import pandas as pd
            
            # Read the CSV file
            df = pd.read_csv(csv_path)
            print(f"[QUOTES DEBUG] Parsing {os.path.basename(csv_path)} for concept '{concept_name}'")
            print(f"[QUOTES DEBUG] CSV has {len(df)} rows and columns: {list(df.columns)}")
            
            # Look for columns that might contain quotes or specific use information
            quote_columns = ['Main Answer', 'Specific Use', 'specific_use', 'Quotes', 'quotes', 'Citation', 'citation', 'Text', 'text']
            
            # First, try exact concept matching
            for col in quote_columns:
                if col in df.columns:
                    print(f"[QUOTES DEBUG] Checking column '{col}' for exact matches")
                    # Filter rows that might contain the concept
                    concept_rows = df[df[col].astype(str).str.contains(concept_name, case=False, na=False)]
                    print(f"[QUOTES DEBUG] Found {len(concept_rows)} rows with exact concept match")
                    
                    for _, row in concept_rows.iterrows():
                        content = str(row[col]).strip()
                        if content and content != 'nan' and len(content) > 10:
                            # If this is Main Answer column, extract specific use sections
                            if col == 'Main Answer':
                                specific_uses = self._extract_specific_uses(content, concept_name)
                                quotes.extend(specific_uses)
                                print(f"[QUOTES DEBUG] Added {len(specific_uses)} specific uses from Main Answer")
                            else:
                                quotes.append(content)
                                print(f"[QUOTES DEBUG] Added quote: {content[:100]}...")
            
            # If no exact matches, try partial matching (extract main concept without parenthetical info)
            if not quotes:
                main_concept = concept_name.split('(')[0].strip()
                print(f"[QUOTES DEBUG] No exact matches, trying partial match with '{main_concept}'")
                
                for col in quote_columns:
                    if col in df.columns:
                        concept_rows = df[df[col].astype(str).str.contains(main_concept, case=False, na=False)]
                        print(f"[QUOTES DEBUG] Found {len(concept_rows)} rows with partial concept match")
                        
                        for _, row in concept_rows.iterrows():
                            quote_text = str(row[col]).strip()
                            if quote_text and quote_text != 'nan' and len(quote_text) > 10:
                                quotes.append(quote_text)
                                print(f"[QUOTES DEBUG] Added partial match quote: {quote_text[:100]}...")
            
            # If still no quotes, try all text columns with partial matching
            if not quotes:
                main_concept = concept_name.split('(')[0].strip()
                print(f"[QUOTES DEBUG] No quote column matches, trying all text columns with '{main_concept}'")
                text_columns = df.select_dtypes(include=['object']).columns
                
                for col in text_columns:
                    concept_rows = df[df[col].astype(str).str.contains(main_concept, case=False, na=False)]
                    print(f"[QUOTES DEBUG] Column '{col}': Found {len(concept_rows)} rows with partial match")
                    
                    for _, row in concept_rows.iterrows():
                        quote_text = str(row[col]).strip()
                        if quote_text and quote_text != 'nan' and len(quote_text) > 20:  # Longer minimum for general text
                            quotes.append(quote_text)
                            print(f"[QUOTES DEBUG] Added text column quote: {quote_text[:100]}...")
            
            # Remove duplicates and limit to reasonable number
            quotes = list(dict.fromkeys(quotes))[:3]  # Keep first 3 unique quotes
            print(f"[QUOTES DEBUG] Final result: {len(quotes)} quotes found")
            
        except Exception as e:
            print(f"[QUOTES ERROR] Error parsing CSV {csv_path}: {e}")
            import traceback
            traceback.print_exc()
        
        return quotes

    def _extract_concepts_from_csv(self, csv_path):
        """Extract concepts from CSV file by parsing the Main Answer column"""
        concepts = set()
        
        try:
            import pandas as pd
            import re
            
            df = pd.read_csv(csv_path)
            
            # Look for Main Answer column
            if 'Main Answer' in df.columns:
                for _, row in df.iterrows():
                    main_answer = str(row['Main Answer'])
                    if main_answer and main_answer != 'nan':
                        # Extract concepts using regex patterns
                        # Look for numbered lists with concepts
                        concept_patterns = [
                            r'\d+\.\s*\*\*([^*]+?)\*\*',  # **Concept** format (non-greedy)
                            r'\d+\.\s*\*\*([^*]+?)\s*\([^)]+\)\*\*',  # **Concept (Greek)** format
                            r'\d+\.\s*([^(]+?)(?:\s*\([^)]+\))?\s*$',  # Numbered concepts
                        ]
                        
                        for pattern in concept_patterns:
                            matches = re.findall(pattern, main_answer, re.MULTILINE)
                            for match in matches:
                                concept = match.strip()
                                if len(concept) > 3 and self._is_valid_concept(concept):  # Filter out very short matches and invalid concepts
                                    concepts.add(concept)
            
            print(f"[QUOTES DEBUG] Extracted {len(concepts)} concepts from {os.path.basename(csv_path)}")
            
        except Exception as e:
            print(f"[QUOTES ERROR] Error extracting concepts from {csv_path}: {e}")
        
        return list(concepts)

    def _extract_specific_uses(self, content, concept_name):
        """Extract specific use sections for a concept from Main Answer content"""
        specific_uses = []
        
        try:
            import re
            
            # Look for the concept in the content and extract the specific use section
            # Pattern: **Concept** ... *Specific Use*: "quote"
            pattern = rf'\*\*{re.escape(concept_name)}\*\*.*?\*Specific Use\*:\s*"([^"]+)"'
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                quote = match.strip()
                if len(quote) > 10:  # Filter out very short quotes
                    specific_uses.append(quote)
            
            # If no specific use found, try a broader pattern
            if not specific_uses:
                # Look for any quoted text near the concept
                pattern = rf'\*\*{re.escape(concept_name)}\*\*.*?"([^"]+)"'
                matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
                
                for match in matches:
                    quote = match.strip()
                    if len(quote) > 20:  # Longer minimum for general quotes
                        specific_uses.append(quote)
            
            print(f"[QUOTES DEBUG] Extracted {len(specific_uses)} specific uses for '{concept_name}'")
            
        except Exception as e:
            print(f"[QUOTES ERROR] Error extracting specific uses: {e}")
        
        return specific_uses

    def _generate_quotes_table(self, valid_folders, folder_concepts, parameter_csv_files):
        """Generate quotes table data structure"""
        quotes_data = {}
        
        # Extract concepts directly from CSV files instead of using folder_concepts
        all_concepts = set()
        
        # Extract concepts from all CSV files
        for folder in valid_folders:
            folder_name = os.path.basename(folder)
            if folder_name in parameter_csv_files:
                for param_key, csv_path in parameter_csv_files[folder_name].items():
                    csv_concepts = self._extract_concepts_from_csv(csv_path)
                    all_concepts.update(csv_concepts)
                    print(f"[QUOTES DEBUG] Found {len(csv_concepts)} concepts in {param_key} CSV for {folder_name}")
        
        all_concepts = sorted(list(all_concepts))
        print(f"[QUOTES] Processing {len(all_concepts)} concepts across {len(valid_folders)} folders")
        print(f"[QUOTES DEBUG] First 5 concepts: {all_concepts[:5]}")
        
        for concept in all_concepts:
            quotes_data[concept] = {}
            
            for folder in valid_folders:
                folder_name = os.path.basename(folder)
                quotes_data[concept][folder_name] = []
                
                # Look for quotes in parameter CSV files for this folder
                if folder_name in parameter_csv_files:
                    print(f"[QUOTES DEBUG] Processing concept '{concept}' in folder '{folder_name}'")
                    for param_key, csv_path in parameter_csv_files[folder_name].items():
                        quotes = self._parse_csv_for_quotes(csv_path, concept)
                        quotes_data[concept][folder_name].extend(quotes)
                        print(f"[QUOTES DEBUG] Added {len(quotes)} quotes from {param_key} CSV")
                    
                    # Remove duplicates and limit quotes per folder
                    quotes_data[concept][folder_name] = list(dict.fromkeys(quotes_data[concept][folder_name]))[:2]
                    print(f"[QUOTES DEBUG] Final quotes for '{concept}' in '{folder_name}': {len(quotes_data[concept][folder_name])}")
        
        return quotes_data

    def _debug_csv_content(self, csv_path):
        """Debug function to show CSV content structure"""
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            print(f"[QUOTES DEBUG] CSV Structure for {os.path.basename(csv_path)}:")
            print(f"[QUOTES DEBUG] Shape: {df.shape}")
            print(f"[QUOTES DEBUG] Columns: {list(df.columns)}")
            print(f"[QUOTES DEBUG] First few rows:")
            for i, (_, row) in enumerate(df.head(3).iterrows()):
                print(f"[QUOTES DEBUG] Row {i}: {dict(row)}")
        except Exception as e:
            print(f"[QUOTES DEBUG ERROR] Could not read CSV: {e}")

    def _find_source_file(self, parent_folder, folder_name):
        """Find PDF or TXT file for a folder in the parent directory"""
        # Try PDF first
        pdf_path = os.path.join(parent_folder, f"{folder_name}.pdf")
        if os.path.exists(pdf_path):
            return pdf_path, "PDF"
        
        # Try lowercase PDF
        pdf_path_lower = os.path.join(parent_folder, f"{folder_name.lower()}.pdf")
        if os.path.exists(pdf_path_lower):
            return pdf_path_lower, "PDF"
        
        # Try TXT
        txt_path = os.path.join(parent_folder, f"{folder_name}.txt")
        if os.path.exists(txt_path):
            return txt_path, "TXT"
        
        # Try lowercase TXT
        txt_path_lower = os.path.join(parent_folder, f"{folder_name.lower()}.txt")
        if os.path.exists(txt_path_lower):
            return txt_path_lower, "TXT"
        
        return None, None
    
    def _search_quote_in_pdf(self, pdf_path, quote_text):
        """Search for a quote in a PDF file using cached normalized version"""
        normalized_data = self._load_and_normalize_source(pdf_path, "PDF")
        if normalized_data is None:
            return None, False
        
        # Use the same fuzzy matching as TXT (search in combined normalized text)
        if self._fuzzy_match_quote_in_normalized(quote_text, normalized_data):
            # Find which page the quote is on by searching in normalized pages
            clean_quote = self._normalize_text_for_search(quote_text)
            normalized_pages = normalized_data.get('normalized_pages', {})
            
            # Try to find the page number
            for page_num, normalized_page_text in normalized_pages.items():
                if clean_quote.lower() in normalized_page_text.lower():
                    return page_num, True
                # Try aggressive normalization
                agg_quote = self._normalize_text_aggressive(quote_text)
                agg_page = self._normalize_text_aggressive(normalized_page_text)
                if agg_quote.lower() in agg_page.lower():
                    return page_num, True
            
            # If we can't find exact page, estimate from position in combined text
            normalized_text = normalized_data['normalized_text']
            position = normalized_text.lower().find(clean_quote.lower())
            if position == -1:
                position = len(normalized_text) // 2
            
            # Estimate page number based on position
            # Rough estimate: divide position by average chars per page
            avg_chars_per_page = len(normalized_text) / max(1, normalized_data.get('num_pages', 1))
            estimated_page = int(position / avg_chars_per_page) + 1
            estimated_page = min(estimated_page, normalized_data.get('num_pages', 1))
            return estimated_page, True
        
        return None, False
    
    def _normalize_text_for_search(self, text):
        """Normalize text by removing all extra whitespace, newlines, and normalizing spaces"""
        import re
        
        # Remove XML-like tags (e.g., "<the companions>" -> "the companions")
        text = re.sub(r'<([^>]+)>', r'\1', text)
        
        # Remove section markers with brackets (e.g., "[c6]", "[c1]")
        text = re.sub(r'\[c\d+\]', '', text, flags=re.IGNORECASE)
        
        # Remove stray numbers in brackets (e.g., "[13]", "[5]")
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove section markers and special characters (e.g., "§4", "§6", "§3")
        text = re.sub(r'§\d+[a-z]?', '', text)
        
        # Remove standalone forward slashes with spaces (e.g., " / " -> " ")
        text = re.sub(r'\s*/\s*', ' ', text)
        
        # Remove number+letter patterns (e.g., "1095b", "1096a", "1095a", "251095a")
        text = re.sub(r'\b\d{3,}[a-z]\b', '', text)
        
        # Remove page numbers and formatting artifacts
        # Patterns like: "10 15 20 25 30 351157b" (sequences of numbers)
        # Also handle "5 10 15 20 25 30 35" (sequences starting with single digits)
        text = re.sub(r'\b\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+[a-z]?\b', '', text)
        text = re.sub(r'\b\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+[a-z]?\b', '', text)
        # Shorter sequences: "10 15 20 25 30" or "5 10 15 20 25 30"
        text = re.sub(r'\b\d+\s+\d+\s+\d+\s+\d+\s+\d+\b', '', text)
        # Even shorter: "10 15 20" or "5 10 15 20" or "5 10 15 20 25"
        text = re.sub(r'\b\d+\s+\d+\s+\d+\s+\d+\b', '', text)
        # Very short: "5 10 15" or "10 15"
        text = re.sub(r'\b\d+\s+\d+\s+\d+\b', '', text)
        text = re.sub(r'\b\d+\s+\d+\b', '', text)
        
        # File names and paths: "DSHPC081-2_Body_p001-203.indd" or "DSHPC081-2_Body_p001-203. indd"
        # Also handle patterns like "5 DSHPC081-2_Body_p001-203.indd 16922/06/19 3:15 PM 170"
        # Remove number + filename + number + date pattern
        text = re.sub(r'\b\d+\s+[A-Z0-9_-]+\.\s*(indd|pdf|txt|docx?)\s+\d+[/-]\d{1,2}[/-]\d{2,4}\s+\d{1,2}:\s*\d{2}\s*(AM|PM)?\s+\d+\b', '', text, flags=re.IGNORECASE)
        # Remove standalone file names
        text = re.sub(r'\b[A-Z0-9_-]+\.\s*(indd|pdf|txt|docx?)\b', '', text, flags=re.IGNORECASE)
        # Also handle without the dot before extension
        text = re.sub(r'\b[A-Z0-9_-]+\s+(indd|pdf|txt|docx?)\b', '', text, flags=re.IGNORECASE)
        
        # Dates in various formats: "22/06/19 3:15 PM" or "22/06/19 3: 14 PM" (with space in time)
        # Also handle patterns like "16922/06/19" (long number before date)
        text = re.sub(r'\b\d{3,}[/-]\d{1,2}[/-]\d{2,4}\s+\d{1,2}:\s*\d{2}\s*(AM|PM)?\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+\d{1,2}:\s*\d{2}\s*(AM|PM)?\b', '', text, flags=re.IGNORECASE)
        # Also handle dates without time: "22/06/19" or "16922/06/19"
        text = re.sub(r'\b\d{3,}[/-]\d{1,2}[/-]\d{2,4}\b', '', text)
        text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '', text)
        
        # Remove time patterns followed by numbers/letters: ":14 PM 711125b" or ":14 PM 351127a5"
        text = re.sub(r':\s*\d{1,2}\s*(AM|PM)\s+\d+[a-z]?\d*[a-z]?\b', '', text, flags=re.IGNORECASE)
        # Remove numbers/letters followed by time patterns: "711125b :14 PM" or "351127a5 :14 PM"
        text = re.sub(r'\b\d+[a-z]?\d*[a-z]?\s+:\s*\d{1,2}\s*(AM|PM)\b', '', text, flags=re.IGNORECASE)
        # Remove standalone time patterns with spaces: ":14 PM" (when not part of a date)
        text = re.sub(r':\s*\d{1,2}\s*(AM|PM)\b', '', text, flags=re.IGNORECASE)
        
        # Book/chapter markers: "147Book VIII, Chapter 5" or "4Book I, Chapter 4"
        text = re.sub(r'\b\d+Book\s+[IVXLC]+\s*,\s*Chapter\s+\d+\b', '', text, flags=re.IGNORECASE)
        # Also handle without "Chapter": "4Book I"
        text = re.sub(r'\b\d+Book\s+[IVXLC]+\b', '', text, flags=re.IGNORECASE)
        
        # Standalone long number sequences (likely page numbers): "351157b", "41157", "251095a"
        text = re.sub(r'\b\d{5,}[a-z]?\b', '', text)
        
        # Remove patterns like "711125b" (6+ digits + letter) or "351127a5" (6+ digits + letter + digit)
        text = re.sub(r'\b\d{6,}[a-z]\d*[a-z]?\b', '', text)
        
        # Remove standalone single/double/triple digit numbers that are likely page numbers
        # Pattern: space, 1-3 digits, space (but not part of words or dates)
        text = re.sub(r'\s+\d{1,3}\s+', ' ', text)
        
        # Remove numbers immediately after punctuation: "species.10" -> "species."
        text = re.sub(r'([.,;:!?])\d{1,3}(?=\s|$|[A-Za-z])', r'\1', text)
        
        # Remove numbers at start of sentences/paragraphs: "10 Base people" -> "Base people"
        text = re.sub(r'^\s*\d+\s+', '', text, flags=re.MULTILINE)
        
        # Remove numbers at end of lines (likely page numbers)
        text = re.sub(r'\s+\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove numbers attached directly to words (e.g., "nothing5" -> "nothing", "word10" -> "word")
        # Pattern: word ending in letter + 1-3 digits + (optional hyphen + word or end)
        text = re.sub(r'([a-zA-Z])\d{1,3}(?=-|$|\s)', r'\1', text)
        
        # Remove very short words (1-2 letters) after hyphens that are likely formatting artifacts
        # Pattern: word-hyphen-very-short-word at end or before punctuation
        # This handles cases like "nothing5-if" -> "nothing-if" -> "nothing"
        text = re.sub(r'([a-zA-Z]+)-\s*([a-zA-Z]{1,2})(?=[\s.,;:!?]|$)', r'\1', text)
        
        # Remove standalone numbers before words (even if after punctuation): "species.10 Base" -> "species. Base"
        text = re.sub(r'([.,;:!?])\s*\d{1,3}\s+([A-Za-z])', r'\1 \2', text)
        
        # Replace ligatures (common in PDFs): ﬁ → fi, ﬂ → fl, etc.
        ligature_map = {
            'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
            'æ': 'ae', 'œ': 'oe', 'Æ': 'AE', 'Œ': 'OE',
        }
        for ligature, replacement in ligature_map.items():
            text = text.replace(ligature, replacement)
        
        # Replace em dashes (—) and en dashes (–) with hyphens for consistency
        text = text.replace('—', '-').replace('–', '-')
        
        # First, handle hyphenation at line breaks (e.g., "properly-\nspeaking" -> "properly speaking")
        # Remove hyphens that are followed by newline/whitespace and a lowercase letter
        text = re.sub(r'-\s+([a-z])', r'\1', text)
        # Also handle hyphens at end of line followed by newline
        text = re.sub(r'-\s*\n\s*([a-z])', r'\1', text)
        
        # Remove ALL newlines, carriage returns, and other line breaks - make everything continuous
        text = re.sub(r'[\r\n]+', ' ', text)
        
        # Handle hyphens between words:
        # Strategy: Process all word-hyphen-word patterns
        def process_hyphenated_words(match):
            first_part = match.group(1).lower()
            second_part = match.group(2).lower()
            first_part_orig = match.group(1)  # Preserve original case
            second_part_orig = match.group(2)  # Preserve original case
            
            # List of common short words that should NEVER be joined with the following word
            # These are typically prepositions, articles, conjunctions, etc.
            common_short_words = {'for', 'the', 'and', 'but', 'or', 'nor', 'so', 'yet', 'a', 'an', 
                                'in', 'on', 'at', 'to', 'of', 'is', 'it', 'as', 'be', 'by', 'do', 
                                'if', 'my', 'no', 'up', 'we', 'he', 'she', 'me', 'us', 'him', 'her',
                                'his', 'her', 'its', 'our', 'your', 'their', 'this', 'that', 'these', 'those',
                                'with', 'from', 'into', 'onto', 'upon', 'over', 'under', 'above', 'below',
                                'between', 'among', 'through', 'during', 'before', 'after', 'while',
                                'when', 'where', 'why', 'how', 'what', 'who', 'which', 'whom', 'whose'}
            
            # If first part is a common short word, always keep separate
            if first_part in common_short_words:
                return first_part_orig + ' ' + second_part_orig
            
            # If second part is a common short word, always keep separate
            if second_part in common_short_words:
                return first_part_orig + ' ' + second_part_orig
            
            # Check if this looks like a split word (e.g., "intermedi-ate" -> "intermediate")
            # Criteria: 
            # - First part is not a common word (already checked above)
            # - Second part starts with vowel (common in word splits like "intermedi-ate", "gen-eral")
            # - Both parts are relatively short
            # - Combined length is reasonable
            second_starts_vowel = second_part and second_part[0] in 'aeiou'
            
            # Join if it looks like a split word (both parts short, reasonable combined length)
            # The second part starting with a vowel is a strong indicator of a word split
            if (second_starts_vowel and 
                len(first_part) <= 10 and len(second_part) <= 10 and 
                len(first_part) + len(second_part) <= 20):
                # Join the words (preserve case of first letter)
                combined = first_part_orig + second_part_orig
                return combined
            
            # Otherwise, keep as separate words with space
            return first_part_orig + ' ' + second_part_orig
        
        # Match word-hyphen-word patterns (with optional spaces around hyphen)
        text = re.sub(r'\b([a-zA-Z]+)\s*-\s*([a-zA-Z]+)\b', process_hyphenated_words, text)
        
        # Normalize any remaining hyphens with spaces around them (standalone hyphens)
        text = re.sub(r'\s*-\s*', ' ', text)
        
        # Replace all remaining whitespace (spaces, tabs) with single space
        normalized = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        normalized = normalized.strip()
        return normalized
    
    def _normalize_text_aggressive(self, text):
        """More aggressive normalization for flexible matching - handles punctuation differences"""
        import re
        # First normalize whitespace (this handles line breaks and hyphens)
        normalized = self._normalize_text_for_search(text)
        # Remove trailing punctuation that might differ (periods, commas, semicolons)
        normalized = normalized.rstrip('.,;:!?')
        # Normalize spacing around punctuation - ensure single space after punctuation
        normalized = re.sub(r'\s*([,.;:!?])\s*', r'\1 ', normalized)
        # Remove extra spaces that might remain
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized.strip()
    
    def _load_and_normalize_source(self, source_file_path, file_type):
        """Load and normalize a source file, caching the normalized version"""
        # Initialize cache if it doesn't exist
        if not hasattr(self, '_normalized_source_cache'):
            self._normalized_source_cache = {}
        
        # Check cache first
        cache_key = f"{source_file_path}_{file_type}"
        if cache_key in self._normalized_source_cache:
            return self._normalized_source_cache[cache_key]
        
        print(f"[QUOTES DEBUG] Loading and normalizing source: {os.path.basename(source_file_path)}")
        
        normalized_data = None
        
        if file_type == "TXT":
            try:
                with open(source_file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                # Normalize the entire content once
                normalized_data = {
                    'normalized_text': self._normalize_text_for_search(content),
                    'type': 'TXT',
                    'original_length': len(content),
                    'original_text': content  # Keep original for cache file writing
                }
            except Exception as e:
                print(f"[QUOTES DEBUG] Error loading TXT: {e}")
                return None
        
        elif file_type == "PDF":
            # Extract all text from PDF first, then normalize (like TXT)
            all_text = ""
            normalized_pages = {}
            try:
                import pypdf
                with open(source_file_path, 'rb') as file:
                    pdf_reader = pypdf.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages, start=1):
                        page_text = page.extract_text()
                        # Add space between pages to avoid word concatenation
                        all_text += page_text + " "
                        # Also keep page-by-page for page number tracking
                        normalized_pages[page_num] = self._normalize_text_for_search(page_text)
            except ImportError:
                try:
                    from pdfminer.high_level import extract_pages
                    from pdfminer.layout import LTTextContainer
                    for page_num, page_layout in enumerate(extract_pages(source_file_path), start=1):
                        page_text = ""
                        for element in page_layout:
                            if isinstance(element, LTTextContainer):
                                page_text += element.get_text()
                        # Add space between pages
                        all_text += page_text + " "
                        # Also keep page-by-page for page number tracking
                        normalized_pages[page_num] = self._normalize_text_for_search(page_text)
                except Exception as e:
                    print(f"[QUOTES DEBUG] Error loading PDF with pdfminer-six: {e}")
                    return None
            except Exception as e:
                print(f"[QUOTES DEBUG] Error loading PDF with pypdf: {e}")
                return None
            
            # Normalize the entire PDF text as one continuous string (like TXT)
            # This ensures better matching since PDFs often have artificial line breaks
            normalized_text = self._normalize_text_for_search(all_text)
            
            normalized_data = {
                'normalized_pages': normalized_pages,
                'normalized_text': normalized_text,
                'type': 'PDF',
                'num_pages': len(normalized_pages),
                'original_text': all_text  # Keep original for cache file writing
            }
        
        if normalized_data:
            self._normalized_source_cache[cache_key] = normalized_data
            print(f"[QUOTES DEBUG] Cached normalized source: {os.path.basename(source_file_path)} ({len(normalized_data['normalized_text'])} chars)")
            
            # Write cache to TXT file
            self._write_cache_to_file(source_file_path, normalized_data)
        
        return normalized_data
    
    def _write_cache_to_file(self, source_file_path, normalized_data):
        """Write normalized source cache to a TXT file for debugging/review"""
        try:
            # Create output filename: XXX_search.TXT
            base_name = os.path.splitext(os.path.basename(source_file_path))[0]
            output_dir = os.path.dirname(source_file_path)
            output_path = os.path.join(output_dir, f"{base_name}_search.TXT")
            
            # Write normalized text to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"Normalized Search Cache for: {os.path.basename(source_file_path)}\n")
                f.write(f"Generated: {len(normalized_data['normalized_text'])} characters\n")
                f.write("=" * 80 + "\n\n")
                f.write(normalized_data['normalized_text'])
            
            print(f"[QUOTES DEBUG] Wrote cache to: {os.path.basename(output_path)}")
        except Exception as e:
            print(f"[QUOTES DEBUG] Error writing cache file: {e}")
    
    def _fuzzy_match_quote_in_normalized(self, quote_text, normalized_data):
        """Match quote against pre-normalized source data with improved matching"""
        if normalized_data is None:
            return False
        
        normalized_source = normalized_data['normalized_text']
        clean_quote = self._normalize_text_for_search(quote_text)
        
        # Strategy 1: Exact normalized match
        if clean_quote in normalized_source:
            return True
        
        # Strategy 2: Case-insensitive match
        if clean_quote.lower() in normalized_source.lower():
            return True
        
        # Strategy 3: Aggressive normalization (handles punctuation differences)
        agg_quote = self._normalize_text_aggressive(quote_text)
        agg_source = self._normalize_text_aggressive(normalized_source)
        if agg_quote.lower() in agg_source.lower():
            return True
        
        # Strategy 4: Word-based sequential matching (more lenient)
        quote_words = []
        for w in clean_quote.split():
            cleaned = w.strip('.,;:!?()[]{}"\'').lower()
            if len(cleaned) > 1:  # Changed from > 2 to > 1 to include more words
                quote_words.append(cleaned)
        
        source_words = []
        for w in normalized_source.split():
            cleaned = w.strip('.,;:!?()[]{}"\'').lower()
            if len(cleaned) > 1:  # Changed from > 2 to > 1
                source_words.append(cleaned)
        
        if len(quote_words) >= 3:  # Changed from > 5 to >= 3 for shorter quotes
            # Check if we can find the quote words in sequence in the source
            quote_idx = 0
            consecutive_matches = 0
            max_consecutive = 0
            
            for source_word in source_words:
                if quote_idx < len(quote_words) and source_word == quote_words[quote_idx]:
                    quote_idx += 1
                    consecutive_matches += 1
                    max_consecutive = max(max_consecutive, consecutive_matches)
                    if quote_idx == len(quote_words):
                        return True
                else:
                    consecutive_matches = 0
            
            # More lenient matching: if we found at least 70% of words in order (changed from 85%)
            # OR if we have a good consecutive match
            if quote_idx >= len(quote_words) * 0.70:
                return True
            if max_consecutive >= min(3, len(quote_words) * 0.4):  # More lenient consecutive matching
                return True
        
        # Strategy 5: Try without first/last few words (handles prefix/suffix differences like "Quote 1:")
        if len(quote_words) >= 8:  # Changed from > 10 to >= 8
            # Try without first 2 and last 2 words
            middle_quote_words = quote_words[2:-2]
            if len(middle_quote_words) >= 3:  # Changed from > 5 to >= 3
                quote_idx = 0
                for source_word in source_words:
                    if quote_idx < len(middle_quote_words) and source_word == middle_quote_words[quote_idx]:
                        quote_idx += 1
                        if quote_idx == len(middle_quote_words):
                            return True
                # Also try partial match of middle words
                if quote_idx >= len(middle_quote_words) * 0.70:
                    return True
        
        # Strategy 6: Try matching without "Quote X:" prefix if present
        if quote_text.startswith("Quote "):
            quote_without_prefix = quote_text.split(":", 1)
            if len(quote_without_prefix) > 1:
                quote_after_colon = quote_without_prefix[1].strip()
                clean_quote_no_prefix = self._normalize_text_for_search(quote_after_colon)
                if clean_quote_no_prefix.lower() in normalized_source.lower():
                    return True
        
        return False
    
    def _search_quote_in_txt(self, txt_path, quote_text):
        """Search for a quote in a TXT file using cached normalized version"""
        normalized_data = self._load_and_normalize_source(txt_path, "TXT")
        if normalized_data is None:
            return None, False
        
        # Match against normalized source
        if self._fuzzy_match_quote_in_normalized(quote_text, normalized_data):
            # Find position for page estimation
            clean_quote = self._normalize_text_for_search(quote_text)
            normalized_text = normalized_data['normalized_text']
            
            position = normalized_text.lower().find(clean_quote.lower())
            if position == -1:
                position = len(normalized_text) // 2
            
            estimated_page = (position // 2000) + 1
            return estimated_page, True
        
        return None, False
    
    def _verify_quote(self, quote_text, source_file_path, file_type):
        """Verify if a quote exists in the source file"""
        if file_type == "PDF":
            page_num, found = self._search_quote_in_pdf(source_file_path, quote_text)
        elif file_type == "TXT":
            page_num, found = self._search_quote_in_txt(source_file_path, quote_text)
        else:
            return None, False
        
        return page_num, found
    
    def _add_quotes_table_to_doc(self, doc, valid_folders, quotes_data, parent_folder=None):
        """Add quotes table to the document after RAG Parameter Consistency Analysis"""
        try:
            from docx.shared import Inches
            from docx.enum.text import WD_ALIGN_PARAGRAPH
            
            # Debug: Start quote verification analysis
            print(f"[QUOTES DEBUG] ===== Starting Quote Verification Analysis =====")
            if parent_folder:
                print(f"[QUOTES DEBUG] Parent folder: {parent_folder}")
            else:
                print(f"[QUOTES DEBUG] WARNING: No parent folder provided, quote verification will be skipped")
            
            # Initialize normalized source cache
            self._normalized_source_cache = {}
            
            # Find source files for each folder
            folder_source_files = {}
            if parent_folder:
                for folder in valid_folders:
                    folder_name = os.path.basename(folder)
                    source_file, file_type = self._find_source_file(parent_folder, folder_name)
                    if source_file:
                        print(f"[QUOTES DEBUG] ✓ Found {file_type} for folder '{folder_name}': {os.path.basename(source_file)}")
                        print(f"[QUOTES DEBUG]   Full path: {source_file}")
                        folder_source_files[folder_name] = (source_file, file_type)
                    else:
                        print(f"[QUOTES DEBUG] ✗ No source file found for folder '{folder_name}' (tried {folder_name}.pdf, {folder_name.lower()}.pdf, {folder_name}.txt, {folder_name.lower()}.txt)")
            
            # Pre-load and normalize all source files for efficiency
            print(f"[QUOTES DEBUG] Pre-loading and normalizing {len(folder_source_files)} source files...")
            for folder_name, (source_file_path, file_type) in folder_source_files.items():
                normalized_data = self._load_and_normalize_source(source_file_path, file_type)
                if normalized_data is None:
                    print(f"[QUOTES DEBUG] WARNING: Failed to load source file for '{folder_name}'")
            print(f"[QUOTES DEBUG] Finished pre-loading {len(self._normalized_source_cache)} source files")
            
            # Add header for quotes section
            doc.add_heading("Concept Citations and Quotes", level=2)
            doc.add_paragraph(
                "This table shows specific citations and quotes from the original texts for each concept, "
                "extracted from parameter-specific CSV files. Each column represents a different folder/author, "
                "and the quotes show how each concept is used in context. Quotes are color-coded: "
                "green background indicates the quote was found in the source document (with page number), "
                "red background indicates the quote was not found."
            )
            
            # Filter out concepts that have no meaningful quotes
            filtered_quotes_data = {}
            for concept, folder_quotes in quotes_data.items():
                # Check if this concept has any meaningful quotes (not all "No quotes found" or single digits)
                has_meaningful_quotes = False
                for folder_name, folder_data in folder_quotes.items():
                    if isinstance(folder_data, list):
                        quotes = folder_data
                    else:
                        quotes = folder_data.get('quotes', [])
                    
                    # Check if there are any meaningful quotes
                    for quote in quotes:
                        if (isinstance(quote, str) and 
                            quote.strip() and 
                            quote.strip() != "No quotes found" and 
                            not quote.strip().isdigit() and 
                            len(quote.strip()) > 1):
                            has_meaningful_quotes = True
                            break
                    
                    if has_meaningful_quotes:
                        break
                
                # Only include concepts with meaningful quotes
                if has_meaningful_quotes:
                    filtered_quotes_data[concept] = folder_quotes
            
            # If no concepts have meaningful quotes, add a message and return
            if not filtered_quotes_data:
                doc.add_paragraph("No meaningful quotes found for any concepts.")
                return
            
            # Create quotes table
            num_folders = len(valid_folders)
            quotes_table = doc.add_table(rows=1, cols=num_folders + 1)  # +1 for concept column
            quotes_table.style = 'Table Grid'
            
            # Calculate available table width (page width - margins)
            section = doc.sections[0]
            table_width = int(section.page_width - section.left_margin - section.right_margin)
            
            # Set proportional column widths to fit within page width
            # Concept column gets half the width of other columns
            concept_width_ratio = 0.5
            folder_width_ratio = 1.0
            total_ratio = concept_width_ratio + (num_folders * folder_width_ratio)
            
            concept_width = int(table_width * (concept_width_ratio / total_ratio))
            folder_width = int(table_width * (folder_width_ratio / total_ratio))
            
            for i, col in enumerate(quotes_table.columns):
                if i == 0:
                    col.width = concept_width  # Concept column - half size
                else:
                    col.width = folder_width  # Folder columns
            
            # Header row
            header_cells = quotes_table.rows[0].cells
            header_cells[0].text = "Concept"
            
            # Add folder headers with short names
            for i, folder in enumerate(valid_folders):
                folder_name = os.path.basename(folder)
                # Create short folder name (first 8 characters or less)
                short_name = folder_name[:8] if len(folder_name) > 8 else folder_name
                header_cells[i + 1].text = short_name
            
            # Make header bold and set font size
            for cell in quotes_table.rows[0].cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        run.bold = True
                        run.font.size = Inches(0.10)
            
            # Statistics for verification
            verification_stats = {}  # {folder_name: {'found': count, 'not_found': count}}
            
            # Add data rows for each concept
            for concept, folder_quotes in filtered_quotes_data.items():
                row = quotes_table.add_row()
                row.cells[0].text = concept
                
                # Add quotes for each folder
                for i, folder in enumerate(valid_folders):
                    folder_name = os.path.basename(folder)
                    folder_data = folder_quotes.get(folder_name, {})
                    
                    # Handle both old format (list) and new format (dict)
                    if isinstance(folder_data, list):
                        quotes = folder_data
                        sources = []
                    else:
                        quotes = folder_data.get('quotes', [])
                        sources = folder_data.get('sources', [])
                    
                    # Initialize stats for this folder if not exists
                    if folder_name not in verification_stats:
                        verification_stats[folder_name] = {'found': 0, 'not_found': 0}
                    
                    cell = row.cells[i + 1]
                    
                    # Skip verification if "No quotes found"
                    if not quotes or all(q.strip() == "No quotes found" or not q.strip() for q in quotes):
                        cell.text = "No quotes found"
                        # Set font size
                        for para in cell.paragraphs:
                            for run in para.runs:
                                run.font.size = Inches(0.09)
                        continue
                    
                    # Get source file for this folder
                    source_file_path, file_type = folder_source_files.get(folder_name, (None, None))
                    
                    if source_file_path:
                        print(f"[QUOTES DEBUG] Processing folder '{folder_name}' - will search in: {os.path.basename(source_file_path)} ({file_type})")
                    else:
                        print(f"[QUOTES DEBUG] Processing folder '{folder_name}' - no source file available for verification")
                    
                    # Filter out single numeric values and invalid quotes
                    filtered_quotes = []
                    for quote in quotes:
                        quote_stripped = quote.strip()
                        # Skip empty, "No quotes found", or single numeric values
                        if (not quote_stripped or 
                            quote_stripped == "No quotes found" or 
                            (quote_stripped.isdigit() and len(quote_stripped) <= 3)):
                            continue
                        filtered_quotes.append(quote)
                    
                    # If no valid quotes after filtering, mark as no quotes
                    if not filtered_quotes:
                        cell.text = "No quotes found"
                        for para in cell.paragraphs:
                            for run in para.runs:
                                run.font.size = Inches(0.09)
                        continue
                    
                    # Clear cell and add quotes with individual coloring
                    cell.text = ""  # Clear default text
                    
                    # Process each quote individually
                    verified_quotes = []
                    for quote_idx, quote in enumerate(filtered_quotes):
                        print(f"[QUOTES DEBUG]   Quote #{quote_idx + 1} for '{folder_name}': {quote[:50]}...")
                        
                        # Verify quote if source file is available
                        if source_file_path:
                            print(f"[QUOTES DEBUG]     Searching in: {os.path.basename(source_file_path)}")
                            page_num, found = self._verify_quote(quote, source_file_path, file_type)
                            
                            if found:
                                print(f"[QUOTES DEBUG]     ✓ FOUND on page {page_num}")
                                verified_quotes.append((quote, True, page_num))
                                verification_stats[folder_name]['found'] += 1
                            else:
                                print(f"[QUOTES DEBUG]     ✗ NOT FOUND")
                                verified_quotes.append((quote, False, None))
                                verification_stats[folder_name]['not_found'] += 1
                        else:
                            # No source file, mark as not verified
                            verified_quotes.append((quote, None, None))
                    
                    # Add quotes to cell with individual coloring
                    for quote_idx, (quote, found, page_num) in enumerate(verified_quotes):
                        # Add line break before second and subsequent quotes
                        if quote_idx > 0:
                            para = cell.add_paragraph()
                        else:
                            para = cell.paragraphs[0]
                        
                        # Preserve original quote text format (may already have "Quote X:" prefix)
                        quote_text = quote
                        
                        if found is True and page_num:
                            quote_with_page = f"{quote_text} [p.{page_num}]"
                        else:
                            quote_with_page = quote_text
                        
                        run = para.add_run(quote_with_page)
                        run.font.size = Inches(0.09)
                        
                        # Color the run based on verification result
                        from docx.shared import RGBColor
                        if found is True:
                            # Dark green background with white text for contrast
                            run.font.color.rgb = RGBColor(255, 255, 255)  # White text for contrast
                            # Set background color using shading XML
                            rPr = run._element.get_or_add_rPr()
                            shd = OxmlElement('w:shd')
                            shd.set(qn('w:fill'), '006400')  # Dark green background
                            shd.set(qn('w:val'), 'clear')
                            rPr.append(shd)
                        elif found is False:
                            # Red background with white text for contrast
                            run.font.color.rgb = RGBColor(255, 255, 255)  # White text for contrast
                            # Set background color using shading XML
                            rPr = run._element.get_or_add_rPr()
                            shd = OxmlElement('w:shd')
                            shd.set(qn('w:fill'), '8B0000')  # Dark red background
                            shd.set(qn('w:val'), 'clear')
                            rPr.append(shd)
                        # If found is None, no coloring (source file not available)
                    
                    # Add sources if available (add after all quotes)
                    if sources:
                        # Add a blank line before sources
                        para = cell.add_paragraph()
                        sources_label = para.add_run("Sources: ")
                        sources_label.font.size = Inches(0.09)
                        sources_label.bold = True
                        
                        # Add each source on a new line
                        for source_idx, source in enumerate(sources):
                            if source_idx > 0:
                                para = cell.add_paragraph()
                                para.add_run("  ")  # Indent continuation sources
                            source_run = para.add_run(source)
                            source_run.font.size = Inches(0.09)
                    
                    # Truncate if too long (but preserve sources)
                    # Count characters in quotes only, not sources
                    quotes_text_length = sum(len(q) for q, _, _ in verified_quotes)
                    if quotes_text_length > 800:
                        # Truncate quotes but keep sources
                        # Remove quotes paragraphs starting from the end until we're under limit
                        total_length = 0
                        quotes_to_keep = []
                        for quote_data in reversed(verified_quotes):
                            quote, found, page_num = quote_data
                            quote_text = quote  # Preserve original format
                            if found is True and page_num:
                                quote_text = f"{quote_text} [p.{page_num}]"
                            
                            if total_length + len(quote_text) > 800:
                                break
                            quotes_to_keep.insert(0, quote_data)
                            total_length += len(quote_text)
                        
                        # Rebuild cell with truncated quotes
                        cell.text = ""
                        for quote_idx, (quote, found, page_num) in enumerate(quotes_to_keep):
                            if quote_idx > 0:
                                para = cell.add_paragraph()
                            else:
                                para = cell.paragraphs[0]
                            
                            quote_text = quote  # Preserve original format
                            if found is True and page_num:
                                quote_text = f"{quote_text} [p.{page_num}]"
                            
                            run = para.add_run(quote_text)
                            run.font.size = Inches(0.09)
                            
                            # Reapply coloring
                            from docx.shared import RGBColor
                            if found is True:
                                run.font.color.rgb = RGBColor(255, 255, 255)
                                rPr = run._element.get_or_add_rPr()
                                shd = OxmlElement('w:shd')
                                shd.set(qn('w:fill'), '006400')
                                shd.set(qn('w:val'), 'clear')
                                rPr.append(shd)
                            elif found is False:
                                run.font.color.rgb = RGBColor(255, 255, 255)
                                rPr = run._element.get_or_add_rPr()
                                shd = OxmlElement('w:shd')
                                shd.set(qn('w:fill'), '8B0000')
                                shd.set(qn('w:val'), 'clear')
                                rPr.append(shd)
                        
                        # Re-add sources after truncated quotes
                        if sources:
                            para = cell.add_paragraph()
                            sources_label = para.add_run("Sources: ")
                            sources_label.font.size = Inches(0.09)
                            sources_label.bold = True
                            
                            for source_idx, source in enumerate(sources):
                                if source_idx > 0:
                                    para = cell.add_paragraph()
                                    para.add_run("  ")
                                source_run = para.add_run(source)
                                source_run.font.size = Inches(0.09)
                
                # Set font size for concept cell
                for para in row.cells[0].paragraphs:
                    for run in para.runs:
                        run.font.size = Inches(0.09)
            
            # Add summary paragraph
            total_concepts = len(filtered_quotes_data)
            total_quotes = 0
            for folder_quotes in filtered_quotes_data.values():
                for folder_data in folder_quotes.values():
                    if isinstance(folder_data, list):
                        total_quotes += len(folder_data)
                    else:
                        total_quotes += len(folder_data.get('quotes', []))
            
            doc.add_paragraph(
                f"Summary: Found quotes for {total_concepts} concepts with a total of {total_quotes} citations across {num_folders} folders."
            )
            
            # Add verification statistics table
            if verification_stats:
                doc.add_paragraph()  # Add spacing
                doc.add_heading("Quote Verification Statistics", level=3)
                doc.add_paragraph(
                    "This table shows the verification results for quotes found in each source document. "
                    "The hallucination rate indicates the percentage of quotes that were not found in the original texts."
                )
                
                # Create statistics table
                stats_table = doc.add_table(rows=1, cols=5)
                stats_table.style = 'Table Grid'
                
                # Header row
                header_cells = stats_table.rows[0].cells
                header_cells[0].text = "Book/Folder"
                header_cells[1].text = "Found"
                header_cells[2].text = "Not Found"
                header_cells[3].text = "Found %"
                header_cells[4].text = "Hallucination Rate"
                
                # Make header bold
                for cell in header_cells:
                    for para in cell.paragraphs:
                        for run in para.runs:
                            run.bold = True
                            run.font.size = Inches(0.10)
                
                # Add data rows
                for folder_name in sorted(verification_stats.keys()):
                    stats = verification_stats[folder_name]
                    found_count = stats['found']
                    not_found_count = stats['not_found']
                    total_count = found_count + not_found_count
                    
                    if total_count > 0:
                        found_percentage = (found_count / total_count) * 100
                        hallucination_rate = 100 - found_percentage
                    else:
                        found_percentage = 0.0
                        hallucination_rate = 100.0
                    
                    row = stats_table.add_row()
                    row.cells[0].text = folder_name
                    row.cells[1].text = str(found_count)
                    row.cells[2].text = str(not_found_count)
                    row.cells[3].text = f"{found_percentage:.1f}%"
                    row.cells[4].text = f"{hallucination_rate:.1f}%"
                    
                    # Color code the percentage cells
                    from docx.shared import RGBColor
                    # Found % cell - green if > 80%, yellow if 50-80%, red if < 50%
                    found_para = row.cells[3].paragraphs[0]
                    found_run = found_para.runs[0] if found_para.runs else found_para.add_run(row.cells[3].text)
                    if found_percentage >= 80:
                        found_run.font.color.rgb = RGBColor(0, 100, 0)  # Dark green
                    elif found_percentage >= 50:
                        found_run.font.color.rgb = RGBColor(184, 134, 11)  # Dark goldenrod
                    else:
                        found_run.font.color.rgb = RGBColor(139, 0, 0)  # Dark red
                    
                    # Hallucination rate cell - red if > 20%, yellow if 10-20%, green if < 10%
                    hall_para = row.cells[4].paragraphs[0]
                    hall_run = hall_para.runs[0] if hall_para.runs else hall_para.add_run(row.cells[4].text)
                    if hallucination_rate > 20:
                        hall_run.font.color.rgb = RGBColor(139, 0, 0)  # Dark red
                    elif hallucination_rate > 10:
                        hall_run.font.color.rgb = RGBColor(184, 134, 11)  # Dark goldenrod
                    else:
                        hall_run.font.color.rgb = RGBColor(0, 100, 0)  # Dark green
                    
                    # Set font size
                    for cell in row.cells:
                        for para in cell.paragraphs:
                            for run in para.runs:
                                run.font.size = Inches(0.09)
                
                print(f"[QUOTES DEBUG] ===== Quote Verification Statistics =====")
                for folder_name, stats in verification_stats.items():
                    total = stats['found'] + stats['not_found']
                    if total > 0:
                        pct = (stats['found'] / total) * 100
                        print(f"[QUOTES DEBUG] {folder_name}: {stats['found']}/{total} found ({pct:.1f}%), {100-pct:.1f}% hallucination rate")
            
            print(f"[QUOTES] Successfully added quotes table with {total_concepts} concepts and {total_quotes} quotes")
            
        except Exception as e:
            print(f"[QUOTES ERROR] Failed to add quotes table to document: {e}")
            import traceback
            traceback.print_exc()

    def on_llm_grouping_toggle(self):
        # If LLM Grouping is enabled, disable all other grouping controls
        state = 'disabled' if self.llm_grouping_var.get() else 'normal'
        self.agg_color_criteria_cb.config(state=state)
        self.agg_min_letters_entry.config(state=state)
        self.agg_group_by_subletters.set(False if self.llm_grouping_var.get() else self.agg_group_by_subletters.get())
        self.agg_group_by_words.set(False if self.llm_grouping_var.get() else self.agg_group_by_words.get())
        self.sim_threshold_entry.config(state=state)
        self.grouping_logic_combo.config(state=state)
        self.agg_group_by_subletters_cb_state = getattr(self, 'agg_group_by_subletters_cb_state', None)
        self.agg_group_by_words_cb_state = getattr(self, 'agg_group_by_words_cb_state', None)
        # Optionally, disable the checkboxes themselves if you want
        # (requires storing references to the checkboxes)
        # self.agg_group_by_subletters_cb.config(state=state)
        # self.agg_group_by_words_cb.config(state=state)

    def on_semantic_viz_toggle(self):
        """Enable/disable t-SNE configuration when Part D is toggled"""
        state = 'normal' if self.semantic_viz_var.get() else 'disabled'
        # Enable/disable only the interactive widgets (Spinbox, Combobox, Checkbutton)
        # Labels don't need to be disabled
        for widget in getattr(self, 'tsne_config_widgets', []):
            try:
                widget.config(state=state)
            except:
                # Some widgets might not support state, skip them
                pass

    def on_consistency_toggle(self):
        """Enable/disable consistency analysis sub-checkboxes"""
        state = 'normal' if self.analyze_consistency_var.get() else 'disabled'
        self.within_param_cb.config(state=state)
        self.cross_param_cb.config(state=state)
        self.sensitivity_cb.config(state=state)
        self.semantic_viz_cb.config(state=state)
        
        # If main checkbox is unchecked, uncheck all sub-checkboxes
        if not self.analyze_consistency_var.get():
            self.within_param_var.set(False)
            self.cross_param_var.set(False)
            self.sensitivity_var.set(False)
            self.semantic_viz_var.set(False)

def replace_bekker_with_greek(concept, bekker_map):
    for bekker, greek in bekker_map.items():
        if bekker in concept:
            # Replace the Bekker number with the Greek concept
            return concept.replace(bekker, greek)
    return concept

def convert_docx_to_pdf_libreoffice(docx_path, output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(docx_path)
    try:
        result = subprocess.run([
            "libreoffice",
            "--headless",
            "--convert-to", "pdf",
            docx_path,
            "--outdir", output_dir
        ], check=True, capture_output=True, text=True)
        print("LibreOffice output:", result.stdout)
        pdf_path = os.path.join(output_dir, os.path.splitext(os.path.basename(docx_path))[0] + ".pdf")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at {pdf_path}")
        return pdf_path
    except Exception as e:
        print("Error converting DOCX to PDF with LibreOffice:", e)
        return None

def extract_groups_from_llm_output(parsed, concepts):
    # If it's a dict, use values
    if isinstance(parsed, dict):
        return list(parsed.values())
    # If it's a list of dicts with 'concepts' key, extract those
    if isinstance(parsed, list) and all(isinstance(g, dict) and 'concepts' in g for g in parsed):
        return [g['concepts'] for g in parsed]
    # If it's a list of lists, use as-is
    if isinstance(parsed, list) and all(isinstance(g, list) for g in parsed):
        return parsed
    # If it's a flat list, treat each as a singleton group
    if isinstance(parsed, list):
        return [[g] for g in parsed]
    # Fallback: treat all as one group
    return [concepts]

def real_llm_grouping(concepts, prompt, model, output_dir=None):
    """Call the real LLM API for grouping. Returns the LLM's raw output and parsed groups."""
    # Prepare the prompt
    prompt_full = (
        prompt.strip() +
        "\nIf the output is too large, split it into multiple JSON arrays or use short group names. Return only JSON, no explanation.\nConcepts:\n" +
        "\n".join(f"- {c}" for c in concepts)
    )
    print("[LLM GROUPING REAL] Model:", model)
    print("[LLM GROUPING REAL] Prompt:\n", prompt_full)
    print("[LLM GROUPING REAL] Concepts:", concepts)
    llm_output = None
    parsed_groups = None
    error = None
    input_tokens = len(prompt_full.split())
    output_tokens = 0
    all_extracted_groups = []
    try:
        # --- Model selection and mapping logic (from advanced_rag.py) ---
        model_map = {
            "gpt-3.5": "gpt-3.5-turbo",
            "gpt-4o": "gpt-4o",
            "gpt-4o mini": "gpt-4o-mini",
            "o1-mini": "gpt-4o-mini",
            "o3-mini": "gpt-4o-mini",
            "mistral": "mistral-small-latest",
            "mistral-api": "mistral-small-latest",
            "meta-llama-3": "meta-llama/Meta-Llama-3-8B-Instruct",
            "remote meta-llama-3": "meta-llama/Meta-Llama-3-8B-Instruct",
            "qwen3": "Qwen/Qwen1.5-7B-Chat",
            "gemini": "gemini-1.5-pro-latest",
            "claude": "claude-3-5-sonnet-20241022",
            # Add more mappings as needed
        }
        # Max tokens per model (based on public docs)
        max_tokens_map = {
            "gpt-3.5": 16385,
            "gpt-4o": 128000,
            "gpt-4o mini": 128000,
            "o1-mini": 128000,
            "o3-mini": 128000,
            "mistral": 128000,
            "mistral-api": 128000,
            "meta-llama-3": 128000,
            "remote meta-llama-3": 128000,
            "qwen3": 128000,
            "gemini": 128000,
            "claude": 200000,
        }
        normalized = model.lower().replace("-api", "").replace("remote ", "").replace(" ", "-")
        print(f"[LLM GROUPING REAL] Original model: {model}")
        print(f"[LLM GROUPING REAL] Normalized model: {normalized}")
        
        model_key = None
        for key in model_map:
            if key in normalized:
                model_key = key
                break
        
        print(f"[LLM GROUPING REAL] Selected model key: {model_key}")
        print(f"[LLM GROUPING REAL] Available model keys: {list(model_map.keys())}")
        
        # Default max tokens
        max_tokens = max_tokens_map.get(model_key, 128000)
        print(f"[LLM GROUPING REAL] Max tokens: {max_tokens}")
        # --- OpenAI GPT Models ---
        if "gpt" in model.lower() or "o1-mini" in model.lower() or "o3-mini" in model.lower():
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                raise RuntimeError("OpenAI API key not available.")
            import openai
            client = openai.OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
                model=model_map.get(model_key, "gpt-3.5-turbo"),
                messages=[{"role": "user", "content": prompt_full}],
                temperature=0.3,
                max_tokens=max_tokens
            )
            llm_output = response.choices[0].message.content
            try:
                import tiktoken
                enc = tiktoken.encoding_for_model(model_map.get(model_key, "gpt-3.5-turbo"))
                input_tokens = len(enc.encode(prompt_full))
                output_tokens = len(enc.encode(llm_output))
            except Exception:
                input_tokens = len(prompt_full.split())
                output_tokens = len(llm_output.split())
        elif "mistral" in model.lower():
            # Use 128000 for Mistral
            api_key = os.environ.get("MISTRAL_API_KEY")
            if not api_key or not Mistral:
                raise RuntimeError("Mistral API or library not available.")
            client = Mistral(api_key=api_key)
            response = client.chat.complete(
                model=model_map.get(model_key, "mistral-small-latest"),
                messages=[{"role": "user", "content": prompt_full}],
                temperature=0.3,
                max_tokens=max_tokens  # 128000
            )
            llm_output = response.choices[0].message.content
            input_tokens = len(prompt_full.split())
            output_tokens = len(llm_output.split())
        elif "llama" in model.lower():
            # Use 128000 for Llama
            hf_token = os.environ.get("HF_API_TOKEN")
            if not hf_token or not InferenceClient:
                raise RuntimeError("HuggingFace Inference API or library not available.")
            client = InferenceClient(token=hf_token, timeout=120)
            llm_output = client.text_generation(
                prompt_full,
                model=model_map.get(model_key, "meta-llama/Meta-Llama-3-8B-Instruct"),
                temperature=0.3,
                max_new_tokens=max_tokens  # 128000
            )
            input_tokens = len(prompt_full.split())
            output_tokens = len(llm_output.split())
        elif "qwen3" in model.lower():
            # Use 128000 for Qwen3
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            model_id = model_map.get(model_key, "Qwen/Qwen1.5-7B-Chat")
            hf_token = os.environ.get("HF_API_TOKEN")
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
            model_qwen = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token)
            pipe = pipeline("text-generation", model=model_qwen, tokenizer=tokenizer)
            result = pipe(prompt_full, max_new_tokens=max_tokens, do_sample=True, temperature=0.3)  # 128000
            llm_output = result[0]["generated_text"]
            input_tokens = len(prompt_full.split())
            output_tokens = len(llm_output.split())
        elif "gemini" in model.lower():
            # Use 128000 for Gemini
            import google.generativeai as genai
            gemini_api_key = os.environ.get("GEMINI_API_KEY")
            if not gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY not set in environment.")
            genai.configure(api_key=gemini_api_key)
            model_name = model_map.get(model_key, "gemini-1.5-pro-latest")
            model_gemini = genai.GenerativeModel(model_name)
            response = model_gemini.generate_content(prompt_full, generation_config={"max_output_tokens": max_tokens})  # 128000
            llm_output = response.text
            input_tokens = len(prompt_full.split())
            output_tokens = len(llm_output.split())
        elif "claude" in model.lower():
            # Use 200000 for Claude
            import anthropic
            claude_api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not claude_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set in environment.")
            client = anthropic.Anthropic(api_key=claude_api_key)
            response = client.messages.create(
                model=model_map.get(model_key, "claude-3-5-sonnet-20241022"),
                max_tokens=200000,  # Claude max
                temperature=0.3,
                messages=[{"role": "user", "content": prompt_full}]
            )
            llm_output = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content)
            input_tokens = len(prompt_full.split())
            output_tokens = len(llm_output.split())
        elif "grok" in model.lower():
            print("[LLM GROUPING REAL] Grok 3 and Grok 4 models are not available via public API.")
            raise RuntimeError("Grok 3 and Grok 4 models are not yet implemented (no public API).")
        elif "phi4" in model.lower():
            # --- Phi4 (Not available) ---
            print("[LLM GROUPING REAL] Phi4 model not yet implemented (no public API).")
            raise RuntimeError("Phi4 model not yet implemented (no public API).")
        elif "meta llama 70b" in model.lower():
            # --- Meta Llama 70B (Not available) ---
            print("[LLM GROUPING REAL] Meta Llama 70B model not yet implemented (no public API).")
            raise RuntimeError("Meta Llama 70B model not yet implemented (no public API).")
        elif "deepseek" in model.lower():
            # --- DeepSeek V3 (Not available) ---
            print("[LLM GROUPING REAL] DeepSeek V3 model not yet implemented (no public API).")
            raise RuntimeError("DeepSeek V3 model not yet implemented (no public API).")
        elif "nebius" in model.lower():
            # --- Nebius Models ---
            nebius_api_key = os.environ.get("NEBIUS_API_KEY")
            if not nebius_api_key:
                raise RuntimeError("NEBIUS_API_KEY not set in environment.")
            import openai
            client = openai.OpenAI(
                api_key=nebius_api_key,
                base_url="https://api.nebius.com/v1"
            )
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",  # Default Nebius model
                messages=[{"role": "user", "content": prompt_full}],
                temperature=0.3,
                max_tokens=max_tokens
            )
            llm_output = response.choices[0].message.content
            input_tokens = len(prompt_full.split())
            output_tokens = len(llm_output.split())
        else:
            print(f"[LLM GROUPING REAL] WARNING: Model '{model}' not recognized, trying fallback...")
            # Try to use the model name directly as a fallback
            if "gpt" in model.lower():
                print("[LLM GROUPING REAL] Attempting OpenAI fallback...")
                try:
                    openai_api_key = os.environ.get("OPENAI_API_KEY")
                    if openai_api_key:
                        import openai
                        client = openai.OpenAI(api_key=openai_api_key)
                        response = client.chat.completions.create(
                            model=model,  # Use the model name directly
                            messages=[{"role": "user", "content": prompt_full}],
                            temperature=0.3,
                            max_tokens=max_tokens
                        )
                        llm_output = response.choices[0].message.content
                        input_tokens = len(prompt_full.split())
                        output_tokens = len(llm_output.split())
                    else:
                        raise RuntimeError("OpenAI API key not available for fallback")
                except Exception as e:
                    raise RuntimeError(f"Fallback failed for model '{model}': {e}")
            else:
                raise RuntimeError(f"Unsupported model: {model}")
        print("[LLM GROUPING REAL] LLM Output:\n", llm_output)
        print(f"[LLM GROUPING REAL] Input tokens: {input_tokens}, Output tokens: {output_tokens}")
        print(f"[LLM GROUPING REAL] LLM Output length: {len(llm_output) if llm_output else 0} characters")
        print(f"[LLM GROUPING REAL] LLM Output type: {type(llm_output)}")
        if llm_output:
            print(f"[LLM GROUPING REAL] First 200 characters: {llm_output[:200]}")
            print(f"[LLM GROUPING REAL] Last 200 characters: {llm_output[-200:]}")
        else:
            print("[LLM GROUPING REAL] WARNING: LLM output is empty or None!")
        # Try to parse JSON from the output
        import re
        print(f"[LLM GROUPING REAL] Attempting to parse JSON from LLM output...")
        print(f"[LLM GROUPING REAL] Looking for JSON patterns in output...")
        
        # Extract all JSON arrays or objects from the output
        json_matches = re.findall(r'\{[\s\S]*?\}|\[[\s\S]*?\]', llm_output)
        print(f"[LLM GROUPING REAL] Found {len(json_matches)} potential JSON matches")
        
        for i, match in enumerate(json_matches):
            print(f"[LLM GROUPING REAL] Processing JSON match {i+1}: {match[:100]}...")
            try:
                obj = json.loads(match)
                print(f"[LLM GROUPING REAL] Successfully parsed JSON match {i+1}")
                # If it's a dict of groups, convert to list of lists
                if isinstance(obj, dict):
                    group_list = list(obj.values())
                    all_extracted_groups.extend(group_list)
                    print(f"[LLM GROUPING REAL] Added {len(group_list)} groups from dict")
                elif isinstance(obj, list):
                    all_extracted_groups.extend(obj)
                    print(f"[LLM GROUPING REAL] Added {len(obj)} groups from list")
            except Exception as e:
                print(f"[LLM GROUPING REAL] JSON parse error in match {i+1}: {e}")
                print(f"[LLM GROUPING REAL] Problematic match content: {match}")
        print(f"[LLM GROUPING REAL] Total extracted groups before processing: {len(all_extracted_groups)}")
        
        if all_extracted_groups:
            print(f"[LLM GROUPING REAL] Processing {len(all_extracted_groups)} extracted groups...")
            parsed_groups = extract_groups_from_llm_output(all_extracted_groups, concepts)
            print(f"[LLM GROUPING REAL] After processing: {len(parsed_groups)} final groups")
        else:
            print("[LLM GROUPING REAL] No groups extracted, trying fallback JSON parse...")
            # Try to parse the whole output as JSON
            try:
                parsed_groups = json.loads(llm_output)
                print(f"[LLM GROUPING REAL] Fallback JSON parse successful: {type(parsed_groups)}")
                parsed_groups = extract_groups_from_llm_output(parsed_groups, concepts)
                print(f"[LLM GROUPING REAL] After fallback processing: {len(parsed_groups)} final groups")
            except Exception as e:
                print(f"[LLM GROUPING REAL] Fallback JSON parse error: {e}")
                print("[LLM GROUPING REAL] Using fallback: all concepts in one group")
                parsed_groups = [concepts]
        # Warn if output is likely truncated
        if len(llm_output) > 18000 or (llm_output and not llm_output.rstrip().endswith(']') and not llm_output.rstrip().endswith('}')):
            print("[LLM GROUPING REAL] WARNING: Output may be truncated or incomplete!")
        # After parsing and flattening, print each group and its concepts
        print("[LLM GROUPING REAL] Parsed groups:")
        for idx, group in enumerate(parsed_groups):
            print(f"  Group {idx+1}: {group}")
    except Exception as e:
        error = str(e)
        print("[LLM GROUPING REAL] ERROR:", error)
        parsed_groups = [concepts]
        llm_output = error
    # Save output to file
    if output_dir:
        try:
            with open(os.path.join(output_dir, "llm_grouping_output.json"), "w", encoding="utf-8") as f:
                json.dump(parsed_groups, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("[LLM GROUPING REAL] Failed to save output:", e)
    print(f"[LLM GROUPING REAL] Parsed {len(parsed_groups)} groups.")
    return parsed_groups

def main():
    root = tk.Tk()
    app = UpSetGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()