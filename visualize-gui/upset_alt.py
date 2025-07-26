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
                        present = any(c in folder_concepts[folder] for c in concepts_in_color)
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
                        present = any(c in folder_concepts[folder] for c in concepts_in_color)
                        row.append(1 if present else 0)
                    group_data.append(row)
            
            # Create DataFrame with concepts (groups) as rows and folders as columns
            # This matches the correct upset plot structure: concepts (rows) vs folders (columns)
            # Each row represents a concept/group, each column represents a folder
            # Cell value: True if the concept is present in that folder
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
            
            # Now create the final DataFrame with concepts (groups) as rows and folders as columns
            # Set the index to group names and columns to short folder names
            df_for_upset = df_groups.set_index('Group')[folder_names]
            df_for_upset.columns = short_folder_names  # Set columns to short folder names
            
            # Convert to boolean for upset plot
            df_for_upset = df_for_upset.astype(bool)
            
            print(f"[UPSET DEBUG] Final DataFrame shape: {df_for_upset.shape}")
            print(f"[UPSET DEBUG] Final DataFrame index (concepts/groups): {df_for_upset.index.tolist()}")
            print(f"[UPSET DEBUG] Final DataFrame columns (folders): {df_for_upset.columns.tolist()}")
            print(f"[UPSET DEBUG] Final DataFrame head:")
            print(df_for_upset.head())
            print(f"[UPSET DEBUG] DataFrame index type: {type(df_for_upset.index)}")
            print(f"[UPSET DEBUG] DataFrame columns type: {type(df_for_upset.columns)}")
            print(f"[UPSET DEBUG] DataFrame dtypes: {df_for_upset.dtypes}")
            
            # Check for duplicate group names
            duplicate_groups = df_for_upset.index[df_for_upset.index.duplicated()].tolist()
            if duplicate_groups:
                print(f"[UPSET DEBUG] WARNING: Found duplicate group names: {duplicate_groups}")
                # Remove duplicates by keeping the first occurrence
                df_for_upset = df_for_upset[~df_for_upset.index.duplicated(keep='first')]
                print(f"[UPSET DEBUG] Removed duplicates. New shape: {df_for_upset.shape}")
            
            # Ensure we have at least 2 columns for upset plot
            if len(df_for_upset.columns) < 2:
                print(f"[UPSET DEBUG] WARNING: Only {len(df_for_upset.columns)} columns. Need at least 2 for upset plot.")
                # Add a dummy column if needed
                df_for_upset['dummy'] = False
                print(f"[UPSET DEBUG] Added dummy column. New shape: {df_for_upset.shape}")
            
            # Create UpSet plot with error handling
            upset_plot_path = None  # Initialize variable
            try:
                # Create upset plot showing which concepts are present in which folders
                # Each row is a concept/group, each column is a folder
                try:
                    # Use from_indicators with just the DataFrame - it will use the columns as indicators
                    upset_data = from_indicators(df_for_upset)
                    print('[UPSET DEBUG] UpSet data created successfully')
                except ValueError as e:
                    # If from_indicators fails with custom index, reset to default index
                    print(f"[UPSET DEBUG] from_indicators failed with custom index: {e}")
                    print("[UPSET DEBUG] Resetting to default index and retrying")
                    df_for_upset_reset = df_for_upset.reset_index(drop=True)
                    upset_data = from_indicators(df_for_upset_reset)
                    print('[UPSET DEBUG] UpSet data created successfully with reset index')
                
                # Create figure that fits page height (landscape orientation)
                # Calculate available height for the plot
                page_height_inches = 8.5  # Landscape page height
                margin_inches = 1.0  # Top and bottom margins
                available_height = page_height_inches - (2 * margin_inches)
                
                fig, axes = plt.subplots(1, 1, figsize=(12, available_height))
                
                # Create UpSet plot with explicit subset_size to avoid non-unique groups error
                upset = UpSet(upset_data, show_counts=True, subset_size='count')
                axes = upset.plot(fig=fig)
                bar_ax = axes['intersections']
                matrix_ax = axes['matrix']
                bars = bar_ax.patches
                upset_index = upset_data.index
                
                # Y-axis shows concepts/groups (the rows)
                # X-axis shows intersections (combinations of folders)
                # Matrix shows which concepts are present in each intersection
                
                # Get the actual number of y-tick locations and set labels accordingly
                y_tick_locations = matrix_ax.get_yticks()
                y_tick_labels = matrix_ax.get_yticklabels()
                
                print(f"[UPSET DEBUG] Y-tick locations: {len(y_tick_locations)}, Y-tick labels: {len(y_tick_labels)}")
                print(f"[UPSET DEBUG] Available concepts: {len(df_for_upset.index)}")
                
                # Set y-axis labels to concept names (group names)
                if len(y_tick_locations) <= len(df_for_upset.index):
                    # Use the concept names as y-axis labels
                    concept_labels = df_for_upset.index.tolist()[:len(y_tick_locations)]
                    matrix_ax.set_yticklabels(concept_labels)
                else:
                    # If we have more tick locations than concepts, use what we have
                    matrix_ax.set_yticklabels(df_for_upset.index.tolist())
                
                # Customize the plot
                plt.title("Concept Groups Overlap Across Folders", fontsize=14, pad=20)
                
                # Save the plot
                upset_plot_path = os.path.join(parent, "groups_upset_plot.png")
                plt.savefig(upset_plot_path, dpi=150, bbox_inches='tight', pad_inches=0.5)
                plt.close()
                
            except Exception as e:
                print(f"[UPSET ERROR] Failed to create UpSet plot: {e}")
                print(f"[UPSET ERROR] Exception type: {type(e)}")
                import traceback
                print(f"[UPSET ERROR] Traceback: {traceback.format_exc()}")                
                # Create a placeholder plot path to avoid UnboundLocalError
                upset_plot_path = os.path.join(parent, "groups_upset_plot.png")
                # Create a simple placeholder plot
                try:
                    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                    ax.text(0.5, 0.5, "UpSet plot generation failed", 
                           ha='center', va='center', transform=ax.transAxes, fontsize=14)
                    ax.set_title("Error: Could not generate UpSet plot")
                    plt.savefig(upset_plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                except:
                    pass  # If even the placeholder fails, we'll handle it gracefully
