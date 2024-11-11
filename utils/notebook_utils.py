import ipywidgets as widgets
from IPython.display import display, clear_output

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_search_widget(entries):
    # Create a text widget for input
    search_box = widgets.Text(
        description='Search:',
        placeholder='Type to search...',
        layout=widgets.Layout(width='300px')
    )
    
    # Create an output widget to display results
    output = widgets.Output()

    def on_search(change):
        # Clear previous output
        with output:
            clear_output()
            search_term = change['new']
            if search_term:
                print(f"Results for '{search_term}':")
                if type(entries) == dict:
                    matches = {key: value for key, value in entries.items() if search_term.lower() in key.lower()}
                    if matches:
                        for key, value in matches.items():
                            print(f"{key}: {value}")
                    else:
                        print("No matching keys found.")
                elif type(entries) == list:
                    matches = [entry for entry in entries if search_term.lower() in entry.lower()]
                    if matches:
                        for match in matches:
                            print(f"{match}")
                    else:
                        print("No matching keys found.")
            else:
                print("Please enter a search term.")

    # Attach the search function to the text widget
    search_box.observe(on_search, names='value')

    # Display the widgets
    display(search_box, output)

def create_dynamic_line_plots(data_list, timebase, general_fig_title = "Battery Simulation Data Overview"):
    """
    Create dynamic line plots using Plotly based on a list of data dictionaries and a timebase dictionary.

    Parameters:
        data_list (list): A list of dictionaries with 'label' and 'data' keys.
        timebase (dict): A dictionary with 'label' and 'data' keys, where 'data' is a list of time values.

    Returns:
        None: Displays the generated plot.
    """
    
    # Extract time data from the provided timebase dictionary
    time_data = timebase.get('data', [])
    
    # Determine number of regions and calculate rows and columns
    num_regions = len(data_list)
    max_cols = 4
    num_cols = min(num_regions, max_cols)
    num_rows = (num_regions + max_cols - 1) // max_cols  # Ceiling division

    # Create subplots
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=[data['label'] for data in data_list],
        vertical_spacing=0.1
    )

    # Add traces for each region
    for idx, data in enumerate(data_list):
        row = idx // num_cols + 1
        col = idx % num_cols + 1
        
        # Add line trace for the data
        fig.add_trace(
            go.Scatter(name=data['label'], x=time_data, y=data['data'], mode='lines'),
            row=row,
            col=col
        )

    # Update layout and axes
    fig.update_layout(
        height=num_rows * 400,
        width=1600,
        title_text=general_fig_title,
        showlegend=True,
    )

    # Update y-axes titles as needed
    for idx in range(num_regions):
        row = idx // num_cols + 1
        col = idx % num_cols + 1
        
        fig.update_yaxes(title_text=data_list[idx].get('label', 'Value'), row=row, col=col)
        fig.update_xaxes(title_text=timebase.get('label', 'Time'), row=row, col=col)

    fig.show()
    
def create_single_fig_line_plot(data_list, timebase, general_fig_title = "Battery Simulation Data Overview"):
    """
    Create a single line plot using Plotly for multiple datasets.

    Parameters:
        data_list (list): A list of dictionaries with 'label' and 'data' keys.
        timebase (dict): A dictionary with 'label' and 'data' keys, where 'data' is a list of time values.

    Returns:
        None: Displays the generated plot.
    """
    
    # Extract time data from the provided timebase dictionary
    time_data = timebase.get('data', [])
    
    # Create figure
    fig = go.Figure()

    # Add traces for each dataset in data_list
    for data in data_list:
        fig.add_trace(
            go.Scatter(
                x=time_data,
                y=data['data'],
                mode='lines',
                name=data['label']
            )
        )

    # Update layout
    fig.update_layout(
        height=400,
        width=1000,
        title=general_fig_title,
        xaxis_title=timebase.get('label', 'Time'),
        yaxis_title="Value",
        showlegend=True
    )

    # Show the figure
    fig.show()