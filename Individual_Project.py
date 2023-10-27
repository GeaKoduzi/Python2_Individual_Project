import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import base64  # Import the base64 module
import matplotlib.pyplot as plt

# STANDARD PAGE CONFIGURATIONS
st.set_page_config(layout='wide')

# SAVE FILE SO YOU CAN EASILY NAVIGATE THROUGH DIFFERENT PAGES
@st.cache_data  # Use @st.cache_data() decorator with allow_output_mutation
def load_data(file):
    return pd.read_csv(file) if file is not None else None

# SIDE BAR
st.sidebar.markdown('# Sales Segmenation Analysis')

# First Function - Upload File
uploaded_file = st.sidebar.file_uploader('Upload Your File in CSV Format.', type=['csv'])

# Initialize df as None
df = None

# Second Button - Radio Button to Navigate across Different Pages
selected_page = st.sidebar.radio('Navigation', ['About this App',
                                                'Total Sales Overview',
                                                'Sales by **Channels**', 
                                                'Sales by **Regions**', 
                                                'Sales by **Product Categories**'])

def modify_data(uploaded_file):
    # Load the CSV data into a DataFrame
    df = pd.read_csv(uploaded_file)
    # Sum the columns from index 2 to 8 for 'Total Sales'
    df['Total Sales'] = df.iloc[:, 2:8].sum(axis=1)
    # Insert a 'CustomerID' column at the beginning
    df.insert(0, 'CustomerID', range(1, len(df) + 1))
    # Directly modify the 'Region' column values
    df['Region'] = df['Region'].replace({
        1: 'Lisbon',
        2: 'Oporto',
        3: 'Other'
    })
    # Directly modify the 'Channel' column values
    df['Channel'] = df['Channel'].replace({
        1: 'Horeca',
        2: 'Retail'
    })
    return df



####################################################################################################################################################################################

# 2) TOTAL SALES OVERVIEW  
if selected_page == 'Total Sales Overview':
    if uploaded_file is None:
        st.warning("Upload the CSV file.")
    else:
        df = load_data(uploaded_file)
        st.header("Total Sales Overview ")
        # Replace channel and region labels in the DataFrame
        st.write('---')

        ###########################################################################################################################################################################

        #SALES DATA OVERVIEW AND KEY INSIGHTS 
        st.markdown('#### Sales Data Overview & Key Insights')

        # Create two columns
        column_1, column_2 = st.columns([1.4, 1.2])

        # Column 1: Original Data Frame
        with column_1:
            st.dataframe(df, height=350)  # Adjust width and height as needed

        # Column 2: Key Metrics
        with column_2:
            # Key Insights 
            total_sales_channels = df['Channel'].nunique()
            total_geographic_regions = df['Region'].nunique()
            total_product_categories = df.iloc[0, 2:].nunique()
            total_customers = len(df)  # Excluding the header row
            total_sales_generated = df.iloc[:, 2:].sum().sum()  # Excluding headings
            overview_table = pd.DataFrame({
                'Metric': ['Total Number of Sales Channels', 'Total Number of Geographic Regions', 'Total Number of Product Categories', 'Total Number of Customers', 'Total Sales'],
                'Value': [total_sales_channels, total_geographic_regions, total_product_categories, total_customers, total_sales_generated]})
            st.dataframe(overview_table,width=500)

            # List of Channels
            channel_mapping= {1:'Horeca', 2:'Retail'}
            st.write(f'**List of Channels:** {", ".join(channel_mapping.values())}')
            
            # List of Geographic Regions
            region_mapping = {1: "Lisbon",2: "Oporto",3: "Other"}
            st.write(f'**List of Regions:** {", ".join(region_mapping.values())}')

            # List of Product Categories
            product_categories = df.columns[3:9].tolist()
            st.write(f'**List of Product Categories:** {", ".join(product_categories)}')

        
        column_1, column_2, column_3 = st.columns(3)

        with column_1: 
            min_revenue = df.iloc[:, 2:8].sum(axis=1).min()
            st.warning(f"Minimum Revenue per Customer: {min_revenue:.2f}")
        with column_2: 
            avg_revenue = df.iloc[:, 2:8].sum(axis=1).mean()
            st.info(f"Average Revenue per Customer: {avg_revenue:.2f}")
        with column_3: 
            max_revenue = df.iloc[:, 2:8].sum(axis=1).max()
            st.success(f"Maximum Revenue per Customer: {max_revenue:.2f}")

        st.write('---')

    ###################################################################################################################################################################################

    # SECTION 2: ANALYSIS OF TOTAL SALES DISTRIBUTION
    st.markdown("#### Exploring Total Sales Distribution")

    df = modify_data(uploaded_file)

    # Create two columns
    column_1, column_2 = st.columns([1, 1])

    # COLUMN 1
    with column_1:
        # Calculate an optimal number of bins for the histogram
        num_bins = 5
        bin_range = np.ceil(df['Total Sales'].max() / 1000) * 1000 - np.floor(df['Total Sales'].min() / 1000) * 1000
        bin_size = bin_range / num_bins  # Calculate bin size

        # Calculate bin edges and round them
        min_revenue_rounded = np.floor(df['Total Sales'].min() / 1000) * 1000
        max_revenue_rounded = np.ceil(df['Total Sales'].max() / 1000) * 1000
        bin_edges = np.arange(min_revenue_rounded, max_revenue_rounded + bin_size + 0.01, bin_size)
        bin_edges = np.round(bin_edges, 2)

        # Create the frequency distribution table
        st.markdown("**Frequency Distribution Table**")

        # Calculate the number of customers in each bin
        hist, bin_edges = np.histogram(df['Total Sales'], bins=bin_edges)
        hist = list(hist)

        # Calculate the percentage of customers in each bin and format them
        total_customers = len(df)
        percentages = [f"{(count / total_customers) * 100:.2f}%" for count in hist]

        # Create a DataFrame for the frequency distribution table
        frequency_table = pd.DataFrame({
            'Min Value': bin_edges[:-1],
            'Max Value': bin_edges[1:],
            'Number of Customers': hist,
            '% of Customers': percentages
        })

        # Display the frequency distribution table
        st.dataframe(frequency_table, width=500)

        # Display summary statistics
        st.markdown("**Summary Statistics**")

        min_value = np.min(df['Total Sales'])
        q1 = np.percentile(df['Total Sales'], 25)
        median = np.median(df['Total Sales'])
        q3 = np.percentile(df['Total Sales'], 75)
        max_value = np.max(df['Total Sales'])

        # Create a DataFrame for the summary statistics
        summary_statistics = pd.DataFrame({
            'Statistic': ['Minimum', 'First Quartile', 'Median', 'Third Quartile', 'Maximum'],
            'Value': [min_value, q1, median, q3, max_value]
        })

        # Display the summary statistics in a DataFrame
        st.dataframe(summary_statistics, width=500)

    # COLUMN 2
    with column_2:
        # Create a histogram using go.Histogram with custom bin settings
        histogram_data = [
            go.Histogram(
                x=df['Total Sales'],
                xbins=dict(
                    start=df['Total Sales'].min(),
                    end=df['Total Sales'].max(),
                    size=bin_size
                ),
                marker=dict(color='orange'),
                name='Total Sales Distribution'
            )
        ]

        layout = go.Layout(
            xaxis=dict(title='Total Sales'),
            yaxis=dict(title='Frequency'),
            barmode='overlay',
            title='Histogram',
            width=600,
            height=300
        )

        fig = go.Figure(data=histogram_data, layout=layout)

        # Display the histogram chart
        st.plotly_chart(fig)

        # Create a box plot using go.Box
        box_plot_data = [
            go.Box(
                x=df['Total Sales'],
                boxpoints='outliers',  # Show outliers
                marker=dict(color='green'),  # Set the color to green
            )
        ]

        layout = go.Layout(
            yaxis=dict(title='Total Sales per Customer'),
            width=600,  # Set the width of the figure (you can change this value)
            height=300,
            title='Box-Plot'
        )

        fig = go.Figure(data=box_plot_data, layout=layout)

        # Display the box plot
        st.plotly_chart(fig)

    st.write('---')

    #################################################################################################################################################################################
    
    # SECTION OF TOP CUSTOMERS

    st.markdown('#### Top and Bottom Customers by Total Sales')
    st.markdown("**Top Customers Analysis**")

    # Create columns
    column_1, column_2 = st.columns([0.7, 1.5])

    with column_1:
        # Create a slider for the number of top customers to analyze
        num_top_customers = st.slider("Select the Number of Top Customers to Analyze", 1, 100, 5)
        
        # Filter and sort the DataFrame based on "Total Sales"
        filtered_df = df
        sort_column = 'Total Sales'
        filtered_df = filtered_df.sort_values(by=sort_column, ascending=False)

        # Download button
        csv_data = filtered_df.head(num_top_customers).to_csv(index=False)
        st.download_button("Download List of Top Customers", csv_data, "top_customers.csv", "text/csv")

    with column_2:
        st.write(filtered_df.head(num_top_customers))


    # SECTION 4 BOTTOM CUSTOMERS
    st.markdown("**Bottom Customers Analysis**")
    # Create columns
    column_1, column_2 = st.columns([0.7, 1.5])

    with column_1:
        # Create a slider for the number of bottom customers to analyze
        num_bottom_customers = st.slider("Select the Number of Bottom Customers to Analyze", 1, 100, 5)
        
        # Filter and sort the DataFrame based on "Total Sales"
        filtered_df = df
        sort_column = 'Total Sales'
        filtered_df = filtered_df.sort_values(by=sort_column, ascending=True)

        # Download button
        csv_data = filtered_df.head(num_bottom_customers).to_csv(index=False)
        st.download_button("Download List of Bottom Customers", csv_data, "bottom_customers.csv", "text/csv")

    with column_2:
        st.write(filtered_df.head(num_bottom_customers))



    
########################################################################################################################################################################################

# 3) SALES BY CHANNELS 

if selected_page == 'Sales by **Channels**':
    if uploaded_file is None:
        st.warning("Please upload the CSV file to proceed.")
    else:
        df = modify_data(uploaded_file)
        st.header("Sales Distribution Across Channels")
        st.write('---')

        # Create THREE columns
        column_1, column_2 = st.columns([1, 1.5])

        with column_1:
            st.markdown('#### Key Sales Metrics')

            # Create a DataFrame to store the results
            channel_results = pd.DataFrame()

            # Group the data by Channel and calculate metrics
            channel_metrics = df.groupby('Channel').agg({'CustomerID': 'count','Total Sales': 'sum'})

            # Rename Customer ID column
            channel_metrics = channel_metrics.rename(columns={'CustomerID': 'Total Number of Customers'})

            # Calculate additional metrics
            channel_metrics['% of Customers'] = (channel_metrics['Total Number of Customers'] / channel_metrics['Total Number of Customers'].sum()) * 100
            channel_metrics['% of Annual Sales'] = (channel_metrics['Total Sales'] / channel_metrics['Total Sales'].sum()) * 100
            channel_metrics['Average Sales per Customer'] = (channel_metrics['Total Sales'] / channel_metrics['Total Number of Customers']).round(2)
            channel_metrics['Min Sales per Customer'] = df.groupby('Channel')['Total Sales'].min().round(2)
            channel_metrics['Max Sales per Customer'] = df.groupby('Channel')['Total Sales'].max().round(2)

            # Reset the index and add Channel names
            channel_metrics.reset_index(inplace=True)

            # Reorder the columns
            channel_metrics = channel_metrics[['Channel', 'Total Number of Customers', '% of Customers', 'Total Sales', '% of Annual Sales',
                                               'Average Sales per Customer', 'Max Sales per Customer', 'Min Sales per Customer']]

            # Format percentage columns with two digits after the decimal point and the percentage symbol
            channel_metrics['% of Customers'] = channel_metrics['% of Customers'].apply(lambda x: f'{x:.2f}%')
            channel_metrics['% of Annual Sales'] = channel_metrics['% of Annual Sales'].apply(lambda x: f'{x:.2f}%')

            # Transpose the DataFrame to have metrics as rows and each channel as a separate column
            channel_metrics = channel_metrics.T

            # Set the first row as the column headers
            channel_metrics.columns = channel_metrics.iloc[0]

            # Drop the first row (it contains the column headers)
            channel_metrics = channel_metrics[1:]

            # Display the results DataFrame
            st.dataframe(channel_metrics,width=800)

            with column_2:

                st.markdown('#### Sales and Customer Distribution (Pie Chart & Bar Chart)')
                
                # Display revenue-related graphs 
                with st.expander("Sales Distribution Across Channels"):
                    st.caption('In the first expander, I have utilized matplotlib exclusively to demonstrate the application of this library')
                    col1_revenue, col2_revenue = st.columns([1, 1])

                    with col1_revenue:
                        # Extract the relevant data for the pie chart
                        selected_channels = ['Horeca', 'Retail']
                        labels = selected_channels
                        values = channel_metrics.loc['Total Sales'][selected_channels]

                        # Create the pie chart with adjusted size (e.g., figsize=(6, 6))
                        fig1, ax1 = plt.subplots(figsize=(3, 3))
                        ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=90,textprops={'fontsize': 7})
                        plt.title('Share of Sales per Channel',fontsize=8)

                        # Display the chart using st.pyplot
                        st.pyplot(fig1)

                    with col2_revenue:
                        # Extract the relevant data for the bar chart
                        selected_channels = ['Horeca', 'Retail']
                        channel_names = selected_channels
                        total_revenue = channel_metrics.loc['Total Sales'][selected_channels]

                        # Create the bar chart with adjusted size (e.g., figsize=(6, 6))
                        fig2, ax2 = plt.subplots(figsize=(3, 3))
                        bars = ax2.bar(channel_names, total_revenue)

                        # Annotate the bars with their values
                        for i, bar in enumerate(bars):
                            height = bar.get_height()
                            ax2.annotate(f'{height}',
                                        xy=(bar.get_x() + bar.get_width() / 2, height),
                                        xytext=(0, 1),  # 3 points vertical offset
                                        textcoords="offset points",
                                        ha='center', va='bottom',fontsize=6)

                        # Rotate x-axis labels for better readability
                        plt.xticks(fontsize=7)
                        plt.yticks(fontsize=7)
                        plt.title('Total Sales per Channel',fontsize=8)

                        # Display the chart using st.pyplot
                        st.pyplot(fig2)

                # Display customer count-related graphs
                with st.expander("Customer Distribution across Channels"):
                    col1_customers, col2_customers = st.columns([0.5, 0.5])

                    with col1_customers:
                        # Pie chart displaying customer distribution by channel
                        fig_pie_customers = px.pie(channel_metrics.T,
                                                names=channel_metrics.columns,
                                                values='Total Number of Customers',
                                                title='Share of Customers per Channel',
                                                width=300,
                                                height=350)
                        st.plotly_chart(fig_pie_customers)

                with col2_customers:
                    # Bar chart showcasing total customer count by channel
                    fig_bar_customers = px.bar(channel_metrics.T,
                                               x=channel_metrics.columns,
                                               y='Total Number of Customers',
                                               title='Nr. of Customers per Channel',
                                               width=300,
                                               height=350)
                    st.plotly_chart(fig_bar_customers)

                column_21,column_22 =st.columns([1 ,1])

                # Calculate the total sales for each channel
                channel_sales = df.groupby('Channel')['Total Sales'].sum()

                # Find the channel with the highest sales
                highest_sales_channel = channel_sales.idxmax()
     
                # Find the channel with the lowest sales
                lowest_sales_channel = channel_sales.idxmin()

                with column_21:

                    # Display the channel with the highest sales in a success box
                    st.success(f'Best Performing Sales Channel: {highest_sales_channel}')

                with column_22:
                    # Display the channel with the lowest sales in a warning box
                    st.warning(f'Worst Perfoming Sales Channel: {lowest_sales_channel}')
        st.write('---')

        ###############################################################################################################################################################################

        st.markdown("#### Analysis of Sales Distribution across Channels (Histogram & Box Plot)")

        # Extract unique values from 'Channel'
        channel_categories = df['Channel'].unique().tolist()

        # Create a dropdown menu to let the user select the desired column (only channels)
        selected_column = st.selectbox("Select a channel to analyze", channel_categories)

        # Create two columns
        column_1, column_2 = st.columns([1, 1])

        # Filter the DataFrame based on the selected channel
        filtered_df = df[df['Channel'] == selected_column]

        # Round the minimum and maximum revenues for the histogram
        min_revenue = filtered_df['Total Sales'].min()
        max_revenue = filtered_df['Total Sales'].max()
        min_revenue_rounded = int(np.floor(min_revenue / 1000) * 1000)
        max_revenue_rounded = int(np.ceil(max_revenue / 1000) * 1000)

        # Calculate an optimal number of bins for the histogram
            # COLUMN 1
        num_bins = 5
        bin_range = np.ceil(df['Total Sales'].max() / 1000) * 1000 - np.floor(df['Total Sales'].min() / 1000) * 1000
        bin_size = bin_range / num_bins  # Calculate bin size

        
        # COLUMN 1
        with column_1: 
            # Create the frequency distribution table
            st.markdown("**Frequency Distribution Table**")

            # Calculate bin edges
            min_revenue_rounded = np.floor(df['Total Sales'].min() / 1000) * 1000
            max_revenue_rounded = np.ceil(df['Total Sales'].max() / 1000) * 1000
            bin_edges = np.arange(min_revenue_rounded, max_revenue_rounded + bin_size + 0.01, bin_size)
            bin_edges = np.round(bin_edges, 2)

            # Calculate the number of customers in each bin
            hist, bin_edges = np.histogram(filtered_df['Total Sales'], bins=bin_edges)
            hist = list(hist)

            # Calculate the percentage of customers in each bin and format them
            total_customers = len(filtered_df)
            percentages = [f"{(count / total_customers) * 100:.2f}%" for count in hist]

            # Create a DataFrame for the frequency distribution table
            frequency_table = pd.DataFrame({
                'Min Value': bin_edges[:-1],
                'Max Value': bin_edges[1:],
                'Number of Customers': hist,
                '% of Customers': percentages
            })

            # Display the frequency distribution table
            st.dataframe(frequency_table, width=500)

            # Display summary statistics
            st.markdown("**Summary Statistics**")

            min_value = np.min(filtered_df['Total Sales'])
            q1 = np.percentile(filtered_df['Total Sales'], 25)
            median = np.median(filtered_df['Total Sales'])
            q3 = np.percentile(filtered_df['Total Sales'], 75)
            max_value = np.max(filtered_df['Total Sales'])

            # Create a DataFrame for the summary statistics
            summary_statistics = pd.DataFrame({
                'Statistic': ['Minimum', 'First Quartile', 'Median', 'Third Quartile', 'Maximum'],
                'Value': [min_value, q1, median, q3, max_value]
            })

            # Display the summary statistics in a DataFrame
            st.dataframe(summary_statistics, width=500)

        # COLUMN 2
        with column_2:
            # Create a histogram using go.Histogram with custom bin settings
            histogram_data = [
                go.Histogram(
                    x=filtered_df['Total Sales'],
                    xbins=dict(
                        start=min_revenue_rounded,
                        end=max_revenue_rounded,
                        size=bin_size
                    ),
                    marker=dict(color='orange'),
                    name='Revenue Distribution'
                )
            ]

            layout = go.Layout(
                    xaxis=dict(title='Total Sales'),
                    yaxis=dict(title='Frequency'),
                    barmode='overlay',
                    title='Histogram',
                    width=600,
                    height=300)
            
            fig = go.Figure(data=histogram_data, layout=layout)

            # Display the histogram chart
            st.plotly_chart(fig)

            # Create a box plot using go.Box
            box_plot_data = [
                go.Box(
                    x=filtered_df['Total Sales'],
                    boxpoints='outliers',  # Show outliers
                    marker=dict(color='green'),  # Set the color to green
                )
            ]

            layout = go.Layout(
                xaxis=dict(title='Revenue per Customer'),
                width=600,  # Set the width of the figure (you can change this value)
                height=300,
                title='Box-Plot'
            )

            fig = go.Figure(data=box_plot_data, layout=layout)

            # Display the box plot
            st.plotly_chart(fig)

        st.write('---')

        ###############################################################################################################################################################################

        st.markdown("#### Top Performing Customers in Each Channel")

        # Create columns
        column_1, column_2 = st.columns([0.7, 1.5])

        with column_1:
            # Create a slider and selectbox for user inputs
            num_top_customers = st.slider("1. Select the Number of Top Customers to Analyze", 1, 100, 5)
            channel_option = st.selectbox("2. Select a Channel", df['Channel'].unique())

            # Filter and sort the DataFrame based on the selected channel_option
            filtered_df = df[df['Channel'] == channel_option]
            filtered_df = filtered_df.sort_values(by='Total Sales', ascending=False).head(num_top_customers)

            # Download button for top customers
            csv_data_top = filtered_df.to_csv(index=False)
            st.download_button("Download List of Top Customers", csv_data_top, "top_customers.csv", "text/csv")

        with column_2:
            st.write(f"Top {num_top_customers} Customers for {channel_option} Channel:")
            st.write(filtered_df)

        st.write('---')

        ###############################################################################################################################################################################

        # SECTION 4 BOTTOM CUSTOMERS

        st.markdown("#### Bottom Performing Customers in Each Channel")
        column_1, column_2 = st.columns([1, 1.5])


        with column_1:
            # Create a slider and selectbox for user inputs
            num_bottom_customers = st.slider("1. Select the Number of Bottom Customers to Analyze", 1, 100, 5)
            channel_option = st.selectbox("2. Select a Channel", df['Channel'].unique(), key='bottom_customers_channel_selectbox')

            # Filter and sort the DataFrame based on the selected channel_option
            filtered_df = df[df['Channel'] == channel_option]
            filtered_df = filtered_df.sort_values(by='Total Sales').head(num_bottom_customers)

            # Download button for bottom customers
            csv_data_bottom = filtered_df.to_csv(index=False)
            st.download_button("Download List of Bottom Customers", csv_data_bottom, "bottom_customers.csv", "text/csv")

        with column_2:
                st.write(f"Bottom {num_bottom_customers} Customers for {channel_option} Channel:")
                st.write(filtered_df)
        st.write('---')


        ###############################################################################################################################################################################
        st.markdown('#### Regional Sale and Customers comparison Across Channels')
        column_1, column_2 = st.columns([1, 1])

        with column_1:
            left_space, center_column, right_space = st.columns([0.2, 6, 0.2])
            with center_column:
                st.write('**Number of Customers per Channel in Each Region**')

                channel_1_counts = df[df['Channel'] == 'Horeca'].groupby('Region').size()
                channel_2_counts = df[df['Channel'] == 'Retail'].groupby('Region').size()

                all_regions = ['Lisbon', 'Oporto', 'Other']

                final_df = pd.DataFrame({
                    'Categories': all_regions,
                    'Horeca': channel_1_counts.values,
                    'Retail': channel_2_counts.values
                })

                st.dataframe(final_df, width=400)

                chart_type = st.selectbox("Select Chart Type", ["Stacked Bar Chart", "Side-by-Side Bar Chart", "Heatmap"], index=1)

                if chart_type == "Stacked Bar Chart":
                    plot_df = final_df.melt(id_vars="Categories", value_vars=["Horeca", "Retail"], 
                                            var_name="Channel", value_name="Number of Customers")

                    fig = px.bar(plot_df, x='Channel', y='Number of Customers', color='Categories', 
                                 title="Stacked Bar Chart", barmode="stack")

                elif chart_type == "Side-by-Side Bar Chart":
                    plot_df = final_df.melt(id_vars="Categories", value_vars=["Horeca", "Retail"], 
                                            var_name="Channel", value_name="Number of Customers")

                    fig = px.bar(plot_df, x='Channel', y='Number of Customers', color='Categories', 
                                 title="Side-by-Side Bar Chart", barmode="group")

                elif chart_type == "Heatmap":
                    heatmap_data = final_df.set_index('Categories')
                    fig = px.imshow(heatmap_data.values.T, x=heatmap_data.index, y=heatmap_data.columns,
                                    color_continuous_scale="Blues", title="Heatmap")

                fig.update_layout(
                    width=550,
                    height=400,
                    legend=dict(orientation="v", yanchor="top", y=0.7, xanchor="left", x=1.05, title='')
                )

                st.plotly_chart(fig)


        with column_2:
            left_space, center_column, right_space = st.columns([0.2, 6, 0.2])
            with center_column:
                st.write('**Total Sales per Channel in Each Region**')

                channel_1_sales = df[df['Channel'] == 'Horeca'].groupby('Region')['Total Sales'].sum()
                channel_2_sales = df[df['Channel'] == 'Retail'].groupby('Region')['Total Sales'].sum()

                all_regions = ['Lisbon', 'Oporto', 'Other']

                final_df = pd.DataFrame({
                    'Categories': all_regions,
                    'Horeca': channel_1_sales.values,
                    'Retail': channel_2_sales.values
                })

                st.dataframe(final_df, width=400)

                chart_type_2 = st.selectbox("Select Chart Type 2", ["Stacked Bar Chart", "Side-by-Side Bar Chart", "Heatmap"], index=1)

                if chart_type_2 == "Stacked Bar Chart":
                    plot_df = final_df.melt(id_vars="Categories", value_vars=["Horeca", "Retail"], 
                                            var_name="Channel", value_name="Total Sales")

                    fig = px.bar(plot_df, x='Channel', y='Total Sales', color='Categories', 
                                 title="Stacked Bar Chart", barmode="stack")

                elif chart_type_2 == "Side-by-Side Bar Chart":
                    plot_df = final_df.melt(id_vars="Categories", value_vars=["Horeca", "Retail"], 
                                            var_name="Channel", value_name="Total Sales")

                    fig = px.bar(plot_df, x='Channel', y='Total Sales', color='Categories', 
                                 title="Side-by-Side Bar Chart", barmode="group")

                elif chart_type_2 == "Heatmap":
                    heatmap_data = final_df.set_index('Categories')
                    fig = px.imshow(heatmap_data.values.T, x=heatmap_data.index, y=heatmap_data.columns,
                                    color_continuous_scale="Blues", title="Heatmap")

                fig.update_layout(
                    width=550,
                    height=400,
                    legend=dict(orientation="v", yanchor="top", y=0.7, xanchor="left", x=1.05, title='')
                )

                st.plotly_chart(fig)
        st.write('---')
        ############################################################################################################

        # Get the product categories from the last 6 columns
        product_categories = df.columns[-7:-1].tolist()

        st.markdown('#### Product Category Sales and Customers comparison Across Channels')
        column_1, column_2 = st.columns([1,1])

        with column_1: 
            left_space, center_column, right_space = st.columns([0.2, 6, 0.2])  # Adjust the ratio as needed

            with center_column:
                st.write('**Number of Customers for Each Channel per Product Category**')

                # Create a dataframe for customer counts per product category and channel
                customer_counts = []
                for cat in product_categories:
                    channel_1_count = df[df['Channel'] == 'Horeca'][cat].astype(bool).sum()
                    channel_2_count = df[df['Channel'] == 'Retail'][cat].astype(bool).sum()
                    customer_counts.append([cat, channel_1_count, channel_2_count])

                customer_df = pd.DataFrame(customer_counts, columns=['Category', 'Horeca', 'Retail'])
                st.dataframe(customer_df, width=500)

                # Dropdown for chart type selection
                chart_type_customers = st.selectbox("Select Chart Type", ["Stacked", "Side-by-Side", "Heatmap"], index=1, key='chart_type_customers')
                barmode_customers = "stack" if chart_type_customers == "Stacked" else "group"

                if chart_type_customers == "Heatmap":
                    # Create a heatmap with categories on the y-axis and blue color palette
                    heatmap_data = customer_df.set_index('Category')
                    fig_customers_heatmap = px.imshow(heatmap_data.values.T, x=heatmap_data.index, y=heatmap_data.columns,
                                                      color_continuous_scale="Blues", title="Customer Counts Heatmap")

                    fig_customers_heatmap.update_layout(
                        width=550,
                        height=400,
                        xaxis_title="Category",
                        yaxis_title="Channel",
                    )

                    st.plotly_chart(fig_customers_heatmap)
                else:
                    # Plotting the bar chart for customer counts
                    plot_df = customer_df.melt(id_vars="Category", value_vars=["Horeca", "Retail"], 
                                               var_name="Channel", value_name="Number of Customers")

                    title_text_customers = f"{chart_type_customers} Bar Chart"
                    fig_customers = px.bar(plot_df, x='Channel', y='Number of Customers', color='Category', 
                                           title=title_text_customers, barmode=barmode_customers)

                    # Update the layout
                    fig_customers.update_layout(
                        width=550,
                        height=400,
                        legend=dict(orientation="v", yanchor="top", y=0.7, xanchor="left", x=1.05, title='')
                    )

                    st.plotly_chart(fig_customers)

        with column_2: 
            left_space, center_column, right_space = st.columns([0.2, 6, 0.2])  # Adjust the ratio as needed

            with center_column:
                st.write('**Revenue in Each Channel per Product Category**')

                # Create a dataframe for revenue per product category and channel
                revenues = []
                for cat in product_categories:
                    channel_1_revenue = df[df['Channel'] == 'Horeca'][cat].sum()
                    channel_2_revenue = df[df['Channel'] == 'Retail'][cat].sum()
                    revenues.append([cat, channel_1_revenue, channel_2_revenue])

                revenue_df = pd.DataFrame(revenues, columns=['Category', 'Horeca', 'Retail'])
                st.dataframe(revenue_df, width=500)

                # Dropdown for chart type selection
                chart_type_revenues = st.selectbox("Select Chart Type for Revenue", ["Stacked Bar Chart", "Side-by-Side Bar Chart", "Heatmap"], index=1, key='chart_type_revenues')
                barmode_revenues = "stack" if chart_type_revenues == "Stacked Bar Chart" else "group"

                if chart_type_revenues == "Heatmap":
                    # Create a heatmap with categories on the y-axis and blue color palette
                    heatmap_data = revenue_df.set_index('Category')
                    fig_revenues_heatmap = px.imshow(heatmap_data.values.T, x=heatmap_data.index, y=heatmap_data.columns,color_continuous_scale="Blues", title="Revenue Heatmap")

                    fig_revenues_heatmap.update_layout(
                        width=550,
                        height=400,
                        xaxis_title="Category",
                        yaxis_title="Channel",
                    )

                    st.plotly_chart(fig_revenues_heatmap)
                else:
                    # Plotting the bar chart for revenue
                    plot_df_revenue = revenue_df.melt(id_vars="Category", value_vars=["Horeca", "Retail"], 
                                                      var_name="Channel", value_name="Total Sales")

                    title_text_revenues = f"{chart_type_revenues} Bar Chart"
                    fig_revenues = px.bar(plot_df_revenue, x='Channel', y='Total Sales', color='Category', 
                                          title=title_text_revenues, barmode=barmode_revenues)

                    # Update the layout
                    fig_revenues.update_layout(
                        width=550,
                        height=400,
                        legend=dict(orientation="v", yanchor="top", y=0.7, xanchor="left", x=1.05, title='')
                    )

                    st.plotly_chart(fig_revenues)


##################################################################################################################################################################################


if selected_page == 'Sales by **Regions**':
    if uploaded_file is None:
        st.warning("Please upload the CSV file to proceed.")
    else:
        df = modify_data(uploaded_file)
        st.header("Sales Distribution Across Channels")
        st.write('---')

        column_1, column_2= st.columns([1,1.5]) 

        ######################################################################################################################################################################################

        with column_1:

            st.markdown('#### Key Metrics by Region')

            # Create a DataFrame to store the results
            region_results = pd.DataFrame()

            # Group the data by Region and calculate metrics
            region_metrics = df.groupby('Region').agg({
                'CustomerID': 'count',
                'Total Sales': 'sum',
            })

            # Rename Customer ID column
            region_metrics = region_metrics.rename(columns={'CustomerID': 'Total Number of Customers'})

            # Calculate additional metrics
            region_metrics['% of Customers'] = (region_metrics['Total Number of Customers'] / region_metrics['Total Number of Customers'].sum()) * 100
            region_metrics['% of Annual Sales'] = (region_metrics['Total Sales'] / region_metrics['Total Sales'].sum()) * 100
            region_metrics['Average Sales per Customer'] = (region_metrics['Total Sales'] / region_metrics['Total Number of Customers']).round(2)
            region_metrics['Min Sales per Customer'] = df.groupby('Region')['Total Sales'].min().round(2)
            region_metrics['Max Sales per Customer'] = df.groupby('Region')['Total Sales'].max().round(2)

            # Reset the index and add Region names
            region_metrics.reset_index(inplace=True)

            # Reorder the columns
            region_metrics = region_metrics[['Region', 'Total Number of Customers', '% of Customers', 'Total Sales', '% of Annual Sales',
                                             'Average Sales per Customer', 'Min Sales per Customer', 'Max Sales per Customer']]

            # Format percentage columns with two digits after the decimal point and the percentage symbol
            region_metrics['% of Customers'] = region_metrics['% of Customers'].apply(lambda x: f'{x:.2f}%')
            region_metrics['% of Annual Sales'] = region_metrics['% of Annual Sales'].apply(lambda x: f'{x:.2f}%')

            # Transpose the DataFrame to have metrics as rows and each region as a separate column
            region_metrics = region_metrics.T

            # Set the first row as the column headers
            region_metrics.columns = region_metrics.iloc[0]

            # Drop the first row (it contains the column headers)
            region_metrics = region_metrics[1:]

            # Display the results DataFrame
            st.dataframe(region_metrics)

        #######################################################################################################################################

        with column_2:

            st.markdown('#### Simple Graphs Representing Key Metrics by Region')
            # Expander for Revenue-related Graphs
            with st.expander("Total and Share of Revenue per Region"):

                # Split the content of the expander into two columns
                col1_revenue, col2_revenue = st.columns([0.5, 0.5])

                with col1_revenue:
                    # Pie chart: Amount of revenue generated per region
                    fig_pie_revenue = px.pie(region_metrics.T,
                                             names=region_metrics.columns,
                                             values='Total Sales',
                                             title='% of Sales per Region',
                                             width=300,
                                             height=350)

                    # Show the chart
                    st.plotly_chart(fig_pie_revenue)

                with col2_revenue:
                    # Bar chart: % of revenue per region
                    fig_bar_revenue = px.bar(region_metrics.T,
                                             x=region_metrics.columns,
                                             y='Total Sales',
                                             title='Total Sales per Region',
                                             width=300,
                                             height=350)

                    st.plotly_chart(fig_bar_revenue)

            # Expander for Customer Count-related Graphs
            with st.expander("Total and Share of Customers per Region"):

                # Split the content of the expander into two columns
                col1_customers, col2_customers = st.columns([0.5, 0.5])

                with col1_customers:
                    # Pie chart: Amount of customers per region
                    fig_pie_customers = px.pie(region_metrics.T,
                                               names=region_metrics.columns,
                                               values='Total Number of Customers',
                                               title='% of Customers per Region',
                                               width=300,
                                               height=350)

                    # Show the chart
                    st.plotly_chart(fig_pie_customers)

                with col2_customers:
                    # Bar chart: Number of customers per region
                    fig_bar_customers = px.bar(region_metrics.T,
                                               x=region_metrics.columns,
                                               y='Total Number of Customers',
                                               title='Total Number of Customers per Region',
                                               width=300,
                                               height=350)
                    st.plotly_chart(fig_bar_customers)

            # Assuming you have a DataFrame 'df' containing the relevant data

            column_1, column_2 = st.columns([1, 1])

            # Calculate the total sales for each region
            region_sales = df.groupby('Region')['Total Sales'].sum()

            # Find the region with the highest sales
            highest_sales_region = region_sales.idxmax()

            # Find the region with the lowest sales
            lowest_sales_region = region_sales.idxmin()


            # Section for Best Performing Region
            with column_1:
                # Display the best-performing region in a success box
                st.success(f'Best Performing Region: {highest_sales_region}')

            # Section for Worst Performing Region
            with column_2:
                # Display the worst-performing region in a warning box
                st.warning(f'Worst Performing Region: {lowest_sales_region}')

        st.write('---')


        ############################################################################################################
        st.markdown('#### Number of Customers and Total Revenue in Each Channel Per Region')
        column_1, column_2 = st.columns([1, 1])

        with column_1: 
            left_space, center_column, right_space = st.columns([0.2, 6, 0.2])
            with center_column:
                st.write('**Number of Customers in each region per Channel**')

                # Proceeding with the steps to create the DataFrame and visualization
                region_1_counts = df[df['Region'] == 'Lisbon'].groupby('Channel').size()
                region_2_counts = df[df['Region'] == 'Oporto'].groupby('Channel').size()
                region_3_counts = df[df['Region'] == 'Other'].groupby('Channel').size()

                all_channels = ['Horeca', 'Retail']

                final_df = pd.DataFrame({
                    'Categories': all_channels,
                    'Lisbon': region_1_counts.values,
                    'Oporto': region_2_counts.values,
                    'Other': region_3_counts.values,
                })

                st.dataframe(final_df, width=400)

                # Dropdown for chart type selection
                chart_type_customers = st.selectbox("Select Chart Type for Customers", ["Stacked Bar Chart", "Side-by-Side Bar Chart", "Heatmap"], index=1, key='chart_type_customers')
                
                if chart_type_customers == "Heatmap":
                    heatmap_data = final_df.set_index('Categories')  # Transpose the DataFrame
                    fig = px.imshow(heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index,  # Swap x and y
                                    color_continuous_scale="Blues", title="Heatmap")
                elif chart_type_customers == "Side-by-Side Bar Chart":
                    plot_df = final_df.melt(id_vars="Categories", value_vars=["Lisbon", "Oporto","Other"], 
                                            var_name="Region", value_name="Number of Customers")
                    fig = px.bar(plot_df, x='Region', y='Number of Customers', color='Categories', 
                                 title="Side-by-Side Bar Chart", barmode="group")
                elif chart_type_customers == "Stacked Bar Chart":
                    plot_df = final_df.melt(id_vars="Categories", value_vars=["Lisbon", "Oporto","Other"],
                                            var_name="Region", value_name="Number of Customers")

                    fig = px.bar(plot_df, x='Region', y='Number of Customers', color='Categories', 
                                 title="Stacked Bar Chart", barmode="stack")

                fig.update_layout(
                    width=550,
                    height=400,
                    legend=dict(orientation="v", yanchor="top", y=0.7, xanchor="left", x=1.05, title='')
                )

                st.plotly_chart(fig)

        with column_2: 
            left_space, center_column, right_space = st.columns([0.2, 6, 0.2])
            with center_column:
                st.write('**Number of Customers in each region per Channel**')

                # Proceeding with the steps to create the DataFrame and visualization
                region_1_sales = df[df['Region'] == 'Lisbon'].groupby('Channel')['Total Sales'].sum()
                region_2_sales = df[df['Region'] == 'Oporto'].groupby('Channel')['Total Sales'].sum()
                region_3_sales = df[df['Region'] == 'Other'].groupby('Channel')['Total Sales'].sum()

                all_channels = ['Horeca', 'Retail']

                final_df = pd.DataFrame({
                    'Categories': all_channels,
                    'Lisbon': region_1_sales.values,
                    'Oporto': region_2_sales.values,
                    'Other': region_3_sales.values,
                })

                st.dataframe(final_df, width=400)

                # Dropdown for chart type selection
                chart_type_customers = st.selectbox("Select Chart Type for Sales", ["Stacked Bar Chart", "Side-by-Side Bar Chart", "Heatmap"], index=1, key='chart_type_sales')
                
                if chart_type_customers == "Heatmap":
                    heatmap_data = final_df.set_index('Categories') 
                    fig = px.imshow(heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index,  # Swap x and y
                                    color_continuous_scale="Blues", title="Heatmap")
                elif chart_type_customers == "Side-by-Side Bar Chart":
                    plot_df = final_df.melt(id_vars="Categories", value_vars=["Lisbon", "Oporto","Other"], 
                                            var_name="Region", value_name="Total Sales")
                    fig = px.bar(plot_df, x='Region', y='Total Sales', color='Categories', 
                                 title="Side-by-Side Bar Chart", barmode="group")
                elif chart_type_customers == "Stacked Bar Chart":
                    plot_df = final_df.melt(id_vars="Categories", value_vars=["Lisbon", "Oporto","Other"],
                                            var_name="Region", value_name="Total Sales")
                    fig = px.bar(plot_df, x='Region', y='Total Sales', color='Categories', 
                                 title="Stacked Bar Chart", barmode="stack")
                fig.update_layout(
                    width=550,
                    height=400,
                    legend=dict(orientation="v", yanchor="top", y=0.7, xanchor="left", x=1.05, title='')
                )

                st.plotly_chart(fig)

        st.write('---')

        #########################################################################################################################################

        st.markdown("#### Analysis of Sales Distribution across Channels (Histogram & Box Plot)")

        # Extract unique values from 'Channel'
        region_categories = df['Region'].unique().tolist()

        # Create a dropdown menu to let the user select the desired column (only channels)
        selected_column = st.selectbox("Select a region to analyze", region_categories)

        # Create two columns
        column_1, column_2 = st.columns([1, 1])

        # Filter the DataFrame based on the selected channel
        filtered_df = df[df['Region'] == selected_column]

        # Round the minimum and maximum revenues for the histogram
        min_revenue = filtered_df['Total Sales'].min()
        max_revenue = filtered_df['Total Sales'].max()
        min_revenue_rounded = int(np.floor(min_revenue / 1000) * 1000)
        max_revenue_rounded = int(np.ceil(max_revenue / 1000) * 1000)

        # Calculate an optimal number of bins for the histogram
            # COLUMN 1
        num_bins = 5
        bin_range = np.ceil(df['Total Sales'].max() / 1000) * 1000 - np.floor(df['Total Sales'].min() / 1000) * 1000
        bin_size = bin_range / num_bins  # Calculate bin size

        
        # COLUMN 1
        with column_1: 
            # Create the frequency distribution table
            st.markdown("**Frequency Distribution Table**")

            # Calculate bin edges
            min_revenue_rounded = np.floor(df['Total Sales'].min() / 1000) * 1000
            max_revenue_rounded = np.ceil(df['Total Sales'].max() / 1000) * 1000
            bin_edges = np.arange(min_revenue_rounded, max_revenue_rounded + bin_size + 0.01, bin_size)
            bin_edges = np.round(bin_edges, 2)

            # Calculate the number of customers in each bin
            hist, bin_edges = np.histogram(filtered_df['Total Sales'], bins=bin_edges)
            hist = list(hist)

            # Calculate the percentage of customers in each bin and format them
            total_customers = len(filtered_df)
            percentages = [f"{(count / total_customers) * 100:.2f}%" for count in hist]

            # Create a DataFrame for the frequency distribution table
            frequency_table = pd.DataFrame({
                'Min Value': bin_edges[:-1],
                'Max Value': bin_edges[1:],
                'Number of Customers': hist,
                '% of Customers': percentages
            })

            # Display the frequency distribution table
            st.dataframe(frequency_table, width=500)

            # Display summary statistics
            st.markdown("**Summary Statistics**")

            min_value = np.min(filtered_df['Total Sales'])
            q1 = np.percentile(filtered_df['Total Sales'], 25)
            median = np.median(filtered_df['Total Sales'])
            q3 = np.percentile(filtered_df['Total Sales'], 75)
            max_value = np.max(filtered_df['Total Sales'])

            # Create a DataFrame for the summary statistics
            summary_statistics = pd.DataFrame({
                'Statistic': ['Minimum', 'First Quartile', 'Median', 'Third Quartile', 'Maximum'],
                'Value': [min_value, q1, median, q3, max_value]
            })

            # Display the summary statistics in a DataFrame
            st.dataframe(summary_statistics, width=500)

        # COLUMN 2
        with column_2:
            # Create a histogram using go.Histogram with custom bin settings
            histogram_data = [
                go.Histogram(
                    x=filtered_df['Total Sales'],
                    xbins=dict(
                        start=min_revenue_rounded,
                        end=max_revenue_rounded,
                        size=bin_size
                    ),
                    marker=dict(color='orange'),
                    name='Revenue Distribution'
                )
            ]

            layout = go.Layout(
                    xaxis=dict(title='Total Sales'),
                    yaxis=dict(title='Frequency'),
                    barmode='overlay',
                    title='Histogram',
                    width=600,
                    height=300)
            
            fig = go.Figure(data=histogram_data, layout=layout)

            # Display the histogram chart
            st.plotly_chart(fig)

            # Create a box plot using go.Box
            box_plot_data = [
                go.Box(
                    x=filtered_df['Total Sales'],
                    boxpoints='outliers',  # Show outliers
                    marker=dict(color='green'),  # Set the color to green
                )
            ]

            layout = go.Layout(
                xaxis=dict(title='Revenue per Customer'),
                width=600,  # Set the width of the figure (you can change this value)
                height=300,
                title='Box-Plot'
            )

            fig = go.Figure(data=box_plot_data, layout=layout)

            # Display the box plot
            st.plotly_chart(fig)

        st.write('---')
        ############################################################################################################################
        st.markdown("#### Top Performing Customers in Each Channel")

        # Create columns
        column_1, column_2 = st.columns([0.7, 1.5])

        with column_1:
            # Create a slider and selectbox for user inputs
            num_top_customers = st.slider("1. Select the Number of Top Customers to Analyze", 1, 100, 5)
            region_option = st.selectbox("2. Select a Channel", df['Region'].unique())

            # Filter and sort the DataFrame based on the selected channel_option
            filtered_df = df[df['Region'] == region_option]
            filtered_df = filtered_df.sort_values(by='Total Sales', ascending=False).head(num_top_customers)

            # Download button for top customers
            csv_data_top = filtered_df.to_csv(index=False)
            st.download_button("Download List of Top Customers", csv_data_top, "top_customers.csv", "text/csv")

        with column_2:
            st.write(f"Top {num_top_customers} Customers for {region_option} Region:")
            st.write(filtered_df)

        st.write('---')

        ###############################################################################################################################################################################

        # SECTION 4 BOTTOM CUSTOMERS

        st.markdown("#### Bottom Performing Customers in Each Channel")
        column_1, column_2 = st.columns([1, 1.5])


        with column_1:
            # Create a slider and selectbox for user inputs
            num_bottom_customers = st.slider("1. Select the Number of Bottom Customers to Analyze", 1, 100, 5)
            region_option = st.selectbox("2. Select a Region", df['Region'].unique(), key='bottom_customers_channel_selectbox')

            # Filter and sort the DataFrame based on the selected channel_option
            filtered_df = df[df['Region'] == region_option]
            filtered_df = filtered_df.sort_values(by='Total Sales').head(num_bottom_customers)

            # Download button for bottom customers
            csv_data_bottom = filtered_df.to_csv(index=False)
            st.download_button("Download List of Bottom Customers", csv_data_bottom, "bottom_customers.csv", "text/csv")

        with column_2:
                st.write(f"Bottom {num_bottom_customers} Customers for {region_option} Channel:")
                st.write(filtered_df)
        st.write('---')

        ############################################################################################################################

        # SECTION 3

        # Get the product categories from the last 6 columns
        product_categories = df.columns[-7:-1].tolist()

        # Create two columns
        st.markdown('#### Number of Customers and Total Revenue for Each Product Category Per Region')

        column_1, column_2= st.columns([1,1])

        ##############################################################################################

        with column_1: 
            left_space, center_column, right_space = st.columns([0.1, 8, 0.1])  # Adjust the ratio as needed
            with center_column:
                st.write('**Number of Customers for Each Product Category per Region**')

                # Create a dataframe for customer counts per product category and region
                customer_counts = []
                for cat in product_categories:
                    region_1_count = df[df['Region'] == 'Lisbon'][cat].astype(str).count()
                    region_2_count = df[df['Region'] == 'Oporto'][cat].astype(str).count()
                    region_3_count = df[df['Region'] == 'Other'][cat].astype(str).count()
                    customer_counts.append([cat, region_1_count, region_2_count, region_3_count])

                customer_df = pd.DataFrame(customer_counts, columns=['Category', 'Lisbon', 'Oporto', 'Other'])
                st.dataframe(customer_df, width=500)

                # Dropdown for chart type selection
                chart_type_customers = st.selectbox("Select Chart Type for Customers", ["Stacked", "Side-by-Side","Heatmap"], index=1, key='chart_type_customers3')
                barmode_customers = "stack" if chart_type_customers == "Stacked" else "group"

                if chart_type_customers == "Heatmap":
                    # Create a heatmap with categories on the y-axis and blue color palette
                    heatmap_data = customer_df.set_index('Category')
                    fig_customers_heatmap = px.imshow(heatmap_data.values.T, x=heatmap_data.index, y=heatmap_data.columns,
                                                      color_continuous_scale="Blues", title="Customer Counts Heatmap")

                    fig_customers_heatmap.update_layout(
                        width=550,
                        height=400,
                        xaxis_title="Category",
                        yaxis_title="Product Category",
                    )

                    st.plotly_chart(fig_customers_heatmap)
                else:
                    # Plotting the bar chart for customer counts
                    plot_df = customer_df.melt(id_vars="Category", value_vars=["Lisbon", "Oporto","Other"], 
                                               var_name="Region", value_name="Number of Customers")

                    title_text_customers = f"{chart_type_customers} Bar Chart"
                    fig_customers = px.bar(plot_df, x='Region', y='Number of Customers', color='Category', 
                                           title=title_text_customers, barmode=barmode_customers)

                    # Update the layout
                    fig_customers.update_layout(
                        width=550,
                        height=400,
                        legend=dict(orientation="v", yanchor="top", y=0.7, xanchor="left", x=1.05, title='')
                    )

                    st.plotly_chart(fig_customers)


            with column_2: 
                left_space, center_column, right_space = st.columns([0.1, 8, 0.1])  # Adjust the ratio as needed
                with center_column:
                    st.write('**Total Revenue for Each Product Category per Region**')

                    # Create a dataframe for total revenue per product category and region
                    revenue_totals = []
                    for cat in product_categories:
                        region_1_revenue = df[df['Region'] == 'Lisbon'][cat].sum()
                        region_2_revenue = df[df['Region'] == 'Oporto'][cat].sum()
                        region_3_revenue = df[df['Region'] == 'Other'][cat].sum()
                        revenue_totals.append([cat, region_1_revenue, region_2_revenue, region_3_revenue])

                revenue_df = pd.DataFrame(revenue_totals, columns=['Category', 'Lisbon','Oporto','Other'])
                st.dataframe(revenue_df, width=500)

                # Dropdown for chart type selection
                chart_type_revenues = st.selectbox("Select Chart Type for Revenue", ["Stacked Bar Chart", "Side-by-Side Bar Chart", "Heatmap"], index=1, key='chart_type_revenues')
                barmode_revenues = "stack" if chart_type_revenues == "Stacked Bar Chart" else "group"

                if chart_type_revenues == "Heatmap":
                    # Create a heatmap with categories on the y-axis and blue color palette
                    heatmap_data = revenue_df.set_index('Category')
                    fig_revenues_heatmap = px.imshow(heatmap_data.values.T, x=heatmap_data.index, y=heatmap_data.columns,color_continuous_scale="Blues", title="Revenue Heatmap")

                    fig_revenues_heatmap.update_layout(
                        width=550,
                        height=400,
                        xaxis_title="Category",
                        yaxis_title="Channel",
                    )

                    st.plotly_chart(fig_revenues_heatmap)
                else:
                    # Plotting the bar chart for revenue
                    plot_df_revenue = revenue_df.melt(id_vars="Category", value_vars=["Lisbon", "Oporto",'Other'], 
                                                      var_name="Region", value_name="Total Sales")

                    title_text_revenues = f"{chart_type_revenues} Bar Chart"
                    fig_revenues = px.bar(plot_df_revenue, x='Region', y='Total Sales', color='Category', 
                                          title=title_text_revenues, barmode=barmode_revenues)

                    # Update the layout
                    fig_revenues.update_layout(
                        width=550,
                        height=400,
                        legend=dict(orientation="v", yanchor="top", y=0.7, xanchor="left", x=1.05, title='')
                    )

                    st.plotly_chart(fig_revenues)

#######################################################################################################################################################################################

if selected_page == 'Sales by **Product Categories**':
    if uploaded_file is None:
        st.warning("Upload the CSV file.")
    else:
        df = load_data(uploaded_file)
        df['Total Revenue'] = df.iloc[:, 2:8].sum(axis=1)
        df.insert(0, 'CustomerID', range(1, len(df) + 1))

        # Get the product categories from the last 6 columns
        product_categories = df.columns[-7:-1].tolist()

        # Set a title 
        st.markdown('## Sales Distribution Across Product Categories')
        st.write('---')

        # Displaying metrics for each product category
        column_1, column_2 = st.columns([1, 0.65])
        with column_1:

            # Calculate product category metrics
            st.markdown('#### Key Metrics By Product Category')
            product_metrics = {}
            for category in product_categories:
                total_customers = len(df[df[category] > 0])
                total_revenue = df[category].sum()
                avg_revenue = df[df[category] > 0][category].mean()

                product_metrics[category] = {
                    'Total Customers': total_customers,
                    '% of Customers': f'{(total_customers / len(df)) * 100:.2f}%',
                    'Total Revenue': total_revenue,
                    '% of Annual Revenue': f'{(total_revenue / df["Total Revenue"].sum()) * 100:.2f}%',
                    'Average Revenue per Customer': f'${avg_revenue:.2f}',
                    'Min Revenue': df[df[category] > 0][category].min(),
                    'Max Revenue': df[category].max()
                }

            # Convert dictionary to DataFrame
            st.dataframe(product_metrics)

            # Determining most and least sold product categories
            sales_data = df[product_categories].sum()
            most_sold_category = sales_data.idxmax()
            least_sold_category = sales_data.idxmin()

            st.success(f"**Most Sold Product Category:** {most_sold_category}\nProducts")
            st.warning(f"**Least Sold Product Category:** {least_sold_category}\nProducts")


        with column_2: 
            product_metrics_df = pd.DataFrame(product_metrics).T

            # Radio button to let the user choose the type of chart
            st.markdown('#### Revenue Share per Product Category')
            chart_type = st.radio("Choose Chart Type:", ["Pie Chart", "Bar Chart"],index=1)
            if chart_type == "Pie Chart":
                fig_pie_revenue = px.pie(product_metrics_df,
                                         names=product_metrics_df.index,
                                         values='Total Revenue',
                                         title='% of Revenue by Product Categories',
                                         width=400,
                                         height=400)
                st.plotly_chart(fig_pie_revenue)
            else:  # Bar Chart
                fig_bar_revenue = px.bar(product_metrics_df,
                                         x=product_metrics_df.index,
                                         y='Total Revenue',
                                         title='Total Revenue by Product Categories',
                                         width=400,
                                         height=400)
                st.plotly_chart(fig_bar_revenue)
        st.write('----')
        #########################################################################################################################################
        # Create two columns
        st.markdown('#### Number of Customers and Total Revenue for Each Product Category Per Channel')
        df['Channel'] = df['Channel'].replace({1: 'Horeca', 2: 'Retail'})

        column_1, column_2= st.columns([1,1])

        with column_1:
            left_space, center_column, right_space = st.columns([0.2, 6, 0.2])  # Adjust the ratio as needed
            with center_column:
                st.write('**Number of Customers in each Product Category per Channel**')

                # Assuming channels are defined like ['Online', 'Offline', ...]
                all_channels = df['Channel'].unique()
                
                # Counting the customers for each product category and channel
                customers_by_channel = {}
                for channel in all_channels:
                    channel_data = df[df['Channel'] == channel]
                    customers_by_channel[channel] = (channel_data[product_categories] > 0).sum()

                # Create the customers DataFrame
                customers_df = pd.DataFrame(customers_by_channel)
                
                st.dataframe(customers_df, width=400)

                # Dropdown for chart type selection
                chart_type_customers = st.selectbox("Select Chart Type for Customers", ["Stacked", "Side-by-Side"], index=1, key='chart_type_customers')
                barmode_customers = "stack" if chart_type_customers == "Stacked" else "group"

                # Reshaping the data for plotting
                customers_plot_df = customers_df.reset_index().melt(id_vars="index", value_vars=all_channels, 
                                                                    var_name="Channel", value_name="Number of Customers")
                
                title_text_customers = f"{chart_type_customers} Bar Chart for Number of Customers by Product Categories"
                fig_customers = px.bar(customers_plot_df, x='index', y='Number of Customers', color='Channel', 
                                       title=title_text_customers, barmode=barmode_customers, labels={'index': 'Product Category'})

                fig_customers.update_layout(
                    width=550,
                    height=400,
                    legend=dict(orientation="v", yanchor="top", y=0.7, xanchor="left", x=1.05, title='')
                )

                st.plotly_chart(fig_customers)

        with column_2:

            left_space, center_column, right_space = st.columns([0.2, 6, 0.2])  # Adjust the ratio as needed
            with center_column:
                st.write('**Total Revenue in each Product Category per Channel**')

                # Assuming channels are defined like ['Online', 'Offline', ...]
                all_channels = df['Channel'].unique()
                
                # Summing the revenue for each product category and channel
                revenues_by_channel = {}
                for channel in all_channels:
                    revenues_by_channel[channel] = df[df['Channel'] == channel][product_categories].sum()

                # Create the revenue DataFrame
                revenue_df = pd.DataFrame(revenues_by_channel)
                
                st.dataframe(revenue_df, width=400)

                # Dropdown for chart type selection
                chart_type_revenue = st.selectbox("Select Chart Type for Revenue", ["Stacked", "Side-by-Side"], index=1, key='chart_type_revenue')
                barmode_revenue = "stack" if chart_type_revenue == "Stacked" else "group"

                # Reshaping the data for plotting
                revenue_plot_df = revenue_df.reset_index().melt(id_vars="index", value_vars=all_channels, 
                                                                var_name="Channel", value_name="Total Revenue")
                
                title_text_revenue = f"{chart_type_revenue} Bar Chart for Revenue by Product Categories"
                fig_revenue = px.bar(revenue_plot_df, x='index', y='Total Revenue', color='Channel', 
                                     title=title_text_revenue, barmode=barmode_revenue, labels={'index': 'Product Category'})

                fig_revenue.update_layout(
                    width=550,
                    height=400,
                    legend=dict(orientation="v", yanchor="top", y=0.7, xanchor="left", x=1.05, title='')
                )

                st.plotly_chart(fig_revenue)

        st.write('---')

        ########################################################################################################################################################
        st.markdown('#### Number of Customers and Total Revenue for Each Product Category Per Region')

        all_regions = df['Region'].unique()

        # Create two columns
        column_1, column_2 = st.columns([1,1])

        with column_1:
            left_space, center_column, right_space = st.columns([0.2, 6, 0.2])
            with center_column:
                st.write('**Number of Customers in each Product Category per Region**')
                
                # Counting the customers for each product category and region
                customers_by_region = {}
                for region in all_regions:
                    region_data = df[df['Region'] == region]
                    customers_by_region[region] = (region_data[product_categories] > 0).sum()

                # Create the customers DataFrame
                customers_df = pd.DataFrame(customers_by_region)
                
                st.dataframe(customers_df, width=400)

                # Dropdown for chart type selection
                chart_type_customers = st.selectbox("Select Chart Type for Customers", ["Stacked", "Side-by-Side"], index=1, key='chart_type_customers_selectbox')
                barmode_customers = "stack" if chart_type_customers == "Stacked" else "group"

                # Reshaping the data for plotting
                customers_plot_df = customers_df.reset_index().melt(id_vars="index", value_vars=all_regions, 
                                                                    var_name="Region", value_name="Number of Customers")
                
                title_text_customers = f"{chart_type_customers} Bar Chart for Number of Customers by Product Categories"
                fig_customers = px.bar(customers_plot_df, x='index', y='Number of Customers', color='Region', 
                                       title=title_text_customers, barmode=barmode_customers, labels={'index': 'Product Category'})

                fig_customers.update_layout(width=550, height=400, legend=dict(orientation="v", yanchor="top", y=0.7, xanchor="left", x=1.05, title=''))
                st.plotly_chart(fig_customers)

        with column_2:
            left_space, center_column, right_space = st.columns([0.2, 6, 0.2])
            with center_column:
                st.write('**Total Revenue in each Product Category per Region**')
                
                # Summing the revenue for each product category and region
                revenues_by_region = {}
                for region in all_regions:
                    revenues_by_region[region] = df[df['Region'] == region][product_categories].sum()

                # Create the revenue DataFrame
                revenue_df = pd.DataFrame(revenues_by_region)
                
                st.dataframe(revenue_df, width=400)

                # Dropdown for chart type selection
                chart_type_revenue = st.selectbox("Select Chart Type for Revenue", ["Stacked", "Side-by-Side"], index=1, key='chart_type_revenue_selectbox')
                barmode_revenue = "stack" if chart_type_revenue == "Stacked" else "group"

                # Reshaping the data for plotting
                revenue_plot_df = revenue_df.reset_index().melt(id_vars="index", value_vars=all_regions, 
                                                                var_name="Region", value_name="Total Revenue")
                
                title_text_revenue = f"{chart_type_revenue} Bar Chart for Revenue by Product Categories"
                fig_revenue = px.bar(revenue_plot_df, x='index', y='Total Revenue', color='Region', 
                                     title=title_text_revenue, barmode=barmode_revenue, labels={'index': 'Product Category'})

                fig_revenue.update_layout(width=550, height=400, legend=dict(orientation="v", yanchor="top", y=0.7, xanchor="left", x=1.05, title=''))
                st.plotly_chart(fig_revenue)

        st.write('---')
##########################################################################################################################################################################
        st.markdown('### Product Comparison')

        # User selects products to compare
        products_to_compare = st.multiselect('Select exactly two products to compare', product_categories, key='product_selection')

        if len(products_to_compare) != 2:
            st.warning("Please select exactly two products for comparison.")
        else:
            # Creating three main columns
            col1, col2, col3 = st.columns(3)

            # First Column: Simple Product Comparison
            with col1:
                st.markdown('**Simple Product Comparison**')
                plot_types = ['Histogram', 'Scatter Plot', 'Box Plot']
                selected_plot = st.selectbox('Select Plot Type:', plot_types, index=0, key='simple_comparison_select')
                
                if selected_plot == 'Histogram':
                    # Displaying the histogram as previously created
                    fig_kde = go.Figure()
                    for product in products_to_compare:
                        fig_kde.add_trace(go.Histogram(x=df[product], name=product, opacity=0.6))
                    fig_kde.update_layout(barmode='overlay', title=f"Histogram Comparison", width=400, height=400, 
                                          xaxis_title="Products", 
                                          yaxis_title="Count",
                                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.5, title=''))
                    st.plotly_chart(fig_kde)
                
                elif selected_plot == 'Scatter Plot':
                    fig_scatter = px.scatter(df, x=products_to_compare[0], y=products_to_compare[1], title=f"Scatter Plot: {products_to_compare[0]} vs {products_to_compare[1]}")
                    fig_scatter.update_layout(width=400, height=400)
                    st.plotly_chart(fig_scatter)
                
                else:
                    df_melted = pd.melt(df[products_to_compare], var_name='Product', value_name='Value')
                    fig_box = px.box(df_melted, x='Product', y='Value', title=f"Box Plot Comparison", color='Product')
                    fig_box.update_layout(width=400, height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.6, title=''))
                    st.plotly_chart(fig_box)

            # Second Column: Product Comparison by Channel
            with col2:
                st.markdown('**Product Comparison by Channel**')
                channel_plot_types = ['Sunburst', 'Scatter Plot', 'Heatmap']
                selected_channel_plot = st.selectbox('Select Plot Type:', channel_plot_types, index=0, key='channel_comparison_select')

                if selected_channel_plot == 'Scatter Plot':
                    fig_scatter_channel = px.scatter(df, x=products_to_compare[0], y=products_to_compare[1], color="Channel", title="Scatter Plot by Channel")
                    fig_scatter_channel.update_layout(width=400, height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.5,title=''))
                    st.plotly_chart(fig_scatter_channel)
                
                elif selected_channel_plot == 'Heatmap':
                    heatmap_data_channel = df.groupby("Channel")[products_to_compare].mean().T
                    fig_heatmap_channel = px.imshow(heatmap_data_channel, title="Heatmap by Channel")
                    fig_heatmap_channel.update_yaxes(tickangle=0)
                    fig_heatmap_channel.update_layout(width=400, height=400)
                    st.plotly_chart(fig_heatmap_channel)
                
                else:
                    df_melted = df.melt(id_vars="Channel", value_vars=products_to_compare, var_name="Product", value_name="Value")
                    fig_sunburst_channel = px.sunburst(df_melted, path=["Channel", "Product"], values="Value", title="Sunburst by Channel")
                    fig_sunburst_channel.update_layout(width=400, height=400)
                    st.plotly_chart(fig_sunburst_channel)

            # Third Column: Product Comparison by Region
            with col3:
                st.markdown('**Product Comparison by Region**')
                region_plot_types = ['Heatmap', 'Scatter Plot', 'Sunburst']
                selected_region_plot = st.selectbox('Select Plot Type:', region_plot_types, index=0, key='region_comparison_select')
                
                if selected_region_plot == 'Scatter Plot':
                    fig_scatter_region = px.scatter(df, x=products_to_compare[0], y=products_to_compare[1], color="Region", title="Scatter Plot by Region")
                    fig_scatter_region.update_layout(width=400, height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.5))
                    st.plotly_chart(fig_scatter_region)
                
                elif selected_region_plot == 'Heatmap':
                    heatmap_data_region = df.groupby("Region")[products_to_compare].mean()
                    fig_heatmap_region = px.imshow(heatmap_data_region, title="Heatmap by Region")
                    fig_heatmap_region.update_yaxes(tickangle=0)  # make y-axis labels vertical
                    fig_heatmap_region.update_layout(width=400, height=400)
                    st.plotly_chart(fig_heatmap_region)
                
                else:  # Sunburst
                    df_melted = df.melt(id_vars="Region", value_vars=products_to_compare, var_name="Product", value_name="Value")
                    fig_sunburst_region = px.sunburst(df_melted, path=["Region", "Product"], values="Value", title="Sunburst by Region")
                    fig_sunburst_region.update_layout(width=400, height=400)  # adjust margins to move bar to top
                    st.plotly_chart(fig_sunburst_region)
####################################################################################################################################################################

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Mapping the channel and region values
        df['Channel'] = df['Channel'].replace({1: 'Horeca', 2: 'Retail'})
        df['Region'] = df['Region'].replace({1: 'Lisbon', 2: 'Oporto', 3: 'Other'})

        # Calculate the total revenue
        df['Total Revenue'] = df.iloc[:, 2:].sum(axis=1)

        st.markdown('### Products Comparison Heatmap by Channel')

        # Melt and group the data for the heatmap
        product_categories = df.columns[2:-1].tolist()
        df_melted = df.melt(id_vars=['Channel', 'Region'], value_vars=product_categories, var_name='Product_Category', value_name='Total Revenue')
        grouped_data = df_melted.groupby(['Region', 'Product_Category', 'Channel'])['Total Revenue'].sum().reset_index()

        # Generate heatmap
        channel_1_data = grouped_data[grouped_data['Channel'] == "Horeca"]
        channel_2_data = grouped_data[grouped_data['Channel'] == "Retail"]
        max_value = grouped_data['Total Revenue'].max()
        min_value = grouped_data['Total Revenue'].min()

        trace1 = go.Heatmap(
            x=channel_1_data['Product_Category'],
            y=channel_1_data['Region'],
            z=channel_1_data['Total Revenue'].values.tolist(),
            zmin=min_value,
            zmax=max_value,
            colorscale='Blues',
            showscale=False
        )
        trace2 = go.Heatmap(
            x=channel_2_data['Product_Category'],
            y=channel_2_data['Region'],
            z=channel_2_data['Total Revenue'].values.tolist(),
            zmin=min_value,
            zmax=max_value,
            colorscale='Blues',
            colorbar=dict(thickness=10, len=1, y=0.5)
        )
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Horeca', 'Retail'), shared_xaxes=True)
        fig.append_trace(trace1, 1, 1)
        fig.append_trace(trace2, 2, 1)

        # Increase size and center the heatmap
        fig.update_layout(
            width=1000,  # Adjust margins
            height=500,  # Increase the height
        )

        # Create three columns
        col1, col2, col3 = st.columns([0.5, 4, 0.5])

        # Put the graphs in the middle column
        with col2:
            st.plotly_chart(fig)
    
