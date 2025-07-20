import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import time
import os

# Set page title and configuration
st.set_page_config(
    page_title="Real-Time Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Connect to Cassandra
@st.cache_resource
def get_cassandra_connection():
    try:
        # Configuration to connect to Cassandra
        cluster = Cluster(['127.0.0.1'])  # Replace with your cluster address
        session = cluster.connect('fraud_detection')  # Replace with your keyspace
        return session
    except Exception as e:
        st.error(f"Cannot connect to Cassandra: {e}")
        return None

# Function to fetch data from Cassandra
@st.cache_data(ttl=60)  # Cache data for 60 seconds
def get_transactions(limit=1000, fraud_only=False, time_window=None):
    session = get_cassandra_connection()
    if not session:
        return pd.DataFrame()
    
    # Base query - without any filters initially
    query = "SELECT * FROM transactions_pred"
    
    # Add fraud filter if requested
    if fraud_only:
        query += " WHERE prediction = 1"
    
    # Add LIMIT clause BEFORE ALLOW FILTERING
    query += f" LIMIT {limit}"
    
    # Add ALLOW FILTERING at the very end
    query += " ALLOW FILTERING"
    
    try:
        rows = session.execute(query)
        df = pd.DataFrame(list(rows))
        
        if not df.empty and time_window and 'trans_date_trans_time' in df.columns:
            # Convert dates to datetime format
            df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
            
            # Apply time filter based on your historical data
            start_date = df['trans_date_trans_time'].min()
            end_date = start_date + pd.Timedelta(hours=time_window)
            df = df[(df['trans_date_trans_time'] >= start_date) & 
                    (df['trans_date_trans_time'] <= end_date)]
        
        return df
    except Exception as e:
        st.error(f"Error fetching transactions: {e}")
        return pd.DataFrame()
        
# Function to get statistics
@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_statistics():
    session = get_cassandra_connection()
    if not session:
        return {
            "total_transactions": 0,
            "fraud_transactions": 0,
            "fraud_rate": 0,
            "avg_transaction_amount": 0
        }
    
    try:
        # Use simple queries to avoid problems
        total_query = "SELECT COUNT(*) FROM transactions_pred LIMIT 1000000 ALLOW FILTERING"
        total = session.execute(total_query).one()[0]
        
        # For fraudulent transactions
        fraud_query = "SELECT COUNT(*) FROM transactions_pred WHERE prediction = 1 LIMIT 1000000 ALLOW FILTERING"
        fraud = session.execute(fraud_query).one()[0]
        
        # Calculate fraud rate
        fraud_rate = (fraud / total * 100) if total > 0 else 0
        
        # For average amount
        amt_query = "SELECT amt FROM transactions_pred LIMIT 1000 ALLOW FILTERING"
        amounts = [row.amt for row in session.execute(amt_query)]
        avg_amount = sum(amounts) / len(amounts) if amounts else 0
        
        return {
            "total_transactions": total,
            "fraud_transactions": fraud,
            "fraud_rate": fraud_rate,
            "avg_transaction_amount": avg_amount
        }
    except Exception as e:
        st.error(f"Error getting statistics: {e}")
        return {
            "total_transactions": 0,
            "fraud_transactions": 0,
            "fraud_rate": 0,
            "avg_transaction_amount": 0
        }

# Function to get time series data
@st.cache_data(ttl=60)
def get_time_series_data(hours=24):
    session = get_cassandra_connection()
    if not session:
        return pd.DataFrame()
    
    try:
        # Retrieve all data without time filtering in CQL
        query = "SELECT trans_date_trans_time, prediction FROM transactions_pred LIMIT 1000 ALLOW FILTERING"
        rows = session.execute(query)
        
        # Convert to DataFrame
        df = pd.DataFrame(list(rows))
        if df.empty:
            return pd.DataFrame()
        
        # Convert dates to datetime format
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        
        # Simulate time filtering using historical data
        start_date = df['trans_date_trans_time'].min()
        
        end_date = start_date + pd.Timedelta(hours=hours)
        df = df[(df['trans_date_trans_time'] >= start_date) & 
                (df['trans_date_trans_time'] <= end_date)]
        
        # Group data by hour
        df['hour'] = df['trans_date_trans_time'].dt.floor('H')
        result = df.groupby('hour').agg(
            total_count=('prediction', 'count'),
            fraud_count=('prediction', lambda x: sum(x == 1))
        ).reset_index()
        
        return result
    except Exception as e:
        st.error(f"Error fetching time series data: {e}")
        return pd.DataFrame()

# Function to get fraud distribution by attribute
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_fraud_distribution(attribute):
    session = get_cassandra_connection()
    if not session:
        return pd.DataFrame()
    
    try:
        # Cassandra doesn't have as flexible GROUP BY as SQL
        # We need to retrieve data and do the grouping in Python
        query = f"SELECT {attribute}, prediction FROM transactions_pred LIMIT 5000 ALLOW FILTERING"
        rows = session.execute(query)
        
        # Convert to DataFrame
        df = pd.DataFrame(list(rows))
        if df.empty:
            return pd.DataFrame()
        
        # Group by attribute
        result = df.groupby(attribute).agg(
            total_count=('prediction', 'count'),
            fraud_count=('prediction', lambda x: sum(x == 1))
        ).reset_index()
        
        # Calculate fraud rate
        result['fraud_rate'] = (result['fraud_count'] / result['total_count'] * 100).fillna(0)
        
        return result
    except Exception as e:
        st.error(f"Error fetching distribution data: {e}")
        return pd.DataFrame()

# Function to create amount categories
def get_amount_distribution():
    session = get_cassandra_connection()
    if not session:
        return pd.DataFrame()
    
    try:
        # Retrieve amounts and predictions
        query = "SELECT amt, prediction FROM transactions_pred LIMIT 5000 ALLOW FILTERING"
        rows = session.execute(query)
        
        # Convert to DataFrame
        df = pd.DataFrame(list(rows))
        if df.empty:
            return pd.DataFrame()
        
        # Define amount categories
        df['amount_category'] = pd.cut(
            df['amt'],
            bins=[0, 100, 500, 1000, float('inf')],
            labels=['Less than $100', '$100-$500', '$500-$1000', 'Over $1000']
        )
        
        # Group by category
        result = df.groupby('amount_category').agg(
            total_count=('prediction', 'count'),
            fraud_count=('prediction', lambda x: sum(x == 1))
        ).reset_index()
        
        # Calculate fraud rate
        result['fraud_rate'] = (result['fraud_count'] / result['total_count'] * 100).fillna(0)
        
        return result
    except Exception as e:
        st.error(f"Error fetching amount distribution data: {e}")
        return pd.DataFrame()

# Create sidebar
st.sidebar.title("Fraud Detection System")
try:
    st.sidebar.image("/home/aya/Streaming-Fraud-Detection/images/workflow.jpeg", use_container_width=True)
except Exception:
    st.sidebar.info("Workflow image not found")

# Create menu in sidebar
page = st.sidebar.selectbox(
    "Select Page",
    ["Overview", "Transaction Analysis", "Detailed Statistics"]
)

# Create filter section in sidebar
st.sidebar.header("Filters")
time_filter = st.sidebar.slider(
    "Time window (hours)",
    min_value=1,
    max_value=72,
    value=24
)

fraud_filter = st.sidebar.checkbox("Show only fraudulent transactions")

# Data refresh button
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Display connection status
try:
    session = get_cassandra_connection()
    if session:
        st.sidebar.success("✅ Cassandra Connected")
    else:
        st.sidebar.error("❌ Cassandra Connection Failed")
        
except Exception as e:
    st.sidebar.error(f"Connection error: {e}")

# OVERVIEW PAGE
if page == "Overview":
    st.title("Fraud Detection Overview")
    
    # Display overview metrics
    stats = get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{stats['total_transactions']:,}")
    with col2:
        st.metric("Fraudulent Transactions", f"{stats['fraud_transactions']:,}")
    with col3:
        st.metric("Fraud Rate", f"{stats['fraud_rate']:.2f}%")
    with col4:
        st.metric("Average Amount", f"${stats['avg_transaction_amount']:,.2f}")
    
    # Time series chart
    st.subheader(f"Transaction Trends (Last {time_filter} hours)")
    
    time_data = get_time_series_data(time_filter)
    if not time_data.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_data['hour'],
            y=time_data['total_count'],
            name='All transactions',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=time_data['hour'],
            y=time_data['fraud_count'],
            name='Fraudulent transactions',
            line=dict(color='red', width=2)
        ))
        fig.update_layout(
            xaxis_title='Time',
            yaxis_title='Number of Transactions',
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No time series data available")
    
    # Display recent transactions
    st.subheader("Recent Transactions")
    transactions = get_transactions(limit=10, fraud_only=fraud_filter, time_window=time_filter)
    
    if not transactions.empty:
        # Format data before display
        display_df = transactions.copy()
        
        # Use the "prediction" column instead of "is_fraud" for Cassandra
        if 'prediction' in display_df.columns:
            display_df['Status'] = display_df['prediction'].apply(
                lambda x: "⚠️ FRAUD" if x == 1 else "✅ VALID"
            )
        
        # Select and rename columns for display
        columns_to_display = {
            'transaction_id': 'Transaction ID',
            'trans_date_trans_time': 'Time',
            'cc_num': 'Credit Card Number',
            'amt': 'Amount',
            'merchant': 'Merchant',
            'category': 'Category',
            'Status': 'Status'
        }
        
        # Filter and rename columns that exist
        cols_exist = [col for col in columns_to_display.keys() if col in display_df.columns]
        if cols_exist:
            display_df = display_df[cols_exist].rename(columns={col: columns_to_display[col] for col in cols_exist if col in columns_to_display})
            
            # Highlight fraudulent rows
            def highlight_fraud(row):
                if 'Status' in row and '⚠️ FRAUD' in row['Status']:
                    return ['background-color: rgba(255, 0, 0, 0.2)'] * len(row)
                return [''] * len(row)
            
            st.dataframe(display_df.style.apply(highlight_fraud, axis=1), use_container_width=True)
        else:
            st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No transactions to display")

# TRANSACTION ANALYSIS PAGE
elif page == "Transaction Analysis":
    st.title("Transaction Analysis")
    
    # Create search form
    with st.expander("Search transactions", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            cc_num = st.text_input("Credit Card Number")
            merchant = st.text_input("Merchant")
        with col2:
            amount_min = st.number_input("Minimum Amount", min_value=0.0, value=0.0)
            amount_max = st.number_input("Maximum Amount", min_value=0.0, value=1000000.0)
        
        # Fixed indentation for these elements
        search_fraud = st.checkbox("Search fraudulent transactions only")
        search_button = st.button("Search")
    
    # Search logic - fixed indentation and execution
    if search_button:
        session = get_cassandra_connection()
        if not session:
            st.error("Cannot connect to database")
        else:
            # Search in Cassandra is more limited than in SQL
            # We need to be creative to implement multiple filters
            
            # Start with a basic filter (we'll apply other filters in Python)
            base_query = "SELECT * FROM transactions_pred"
            need_filtering = False
            
            # For credit card or fraud filters, we can use WHERE if the table is indexed
            where_clauses = []
            
            if cc_num:
                where_clauses.append(f"cc_num = '{cc_num}'")
                need_filtering = True
            
            if search_fraud:
                where_clauses.append("prediction = 1")
                need_filtering = True
            
            if where_clauses:
                base_query += " WHERE " + " AND ".join(where_clauses)
            
            # Add LIMIT before ALLOW FILTERING
            base_query += " LIMIT 5000"
            
            # Add ALLOW FILTERING at the end if needed
            if need_filtering:
                base_query += " ALLOW FILTERING"
            
            try:
                rows = session.execute(base_query)
                all_results = pd.DataFrame(list(rows))
                
                # Apply other filters in Python
                filtered_results = all_results
                
                if not filtered_results.empty:
                    if merchant:
                        filtered_results = filtered_results[filtered_results['merchant'].str.contains(merchant, case=False, na=False)]
                    
                    if amount_min > 0:
                        filtered_results = filtered_results[filtered_results['amt'] >= amount_min]
                    
                    if amount_max < 1000000.0:
                        filtered_results = filtered_results[filtered_results['amt'] <= amount_max]
                    
                    # Limit to 100 results for display
                    search_results = filtered_results.head(100)
                    
                    if not search_results.empty:
                        # Add transaction_id if missing
                        if 'transaction_id' not in search_results.columns:
                            search_results['transaction_id'] = search_results.index
                        # Display search results
                        st.subheader(f"Search Results: {len(search_results)} transactions found")
                        
                        # Format display data
                        display_df = search_results.copy()
                        display_df['Status'] = display_df['prediction'].apply(
                            lambda x: "⚠️ FRAUD" if x == 1 else "✅ VALID"
                        )
                        
                        # Display columns
                        columns_to_display = {
                            'transaction_id': 'Transaction ID',
                            'trans_date_trans_time': 'Time',
                            'cc_num': 'Credit Card Number',
                            'amt': 'Amount',
                            'merchant': 'Merchant',
                            'category': 'Category',
                            'Status': 'Status'
                        }
                        
                        cols_exist = [col for col in columns_to_display.keys() if col in display_df.columns]
                        display_df = display_df[cols_exist].rename(columns={col: columns_to_display[col] for col in cols_exist if col in columns_to_display})
                        
                        # Highlight fraudulent rows
                        def highlight_fraud(row):
                            if 'Status' in row and '⚠️ FRAUD' in row['Status']:
                                return ['background-color: rgba(255, 0, 0, 0.2)'] * len(row)
                            return [''] * len(row)
                        
                        st.dataframe(display_df.style.apply(highlight_fraud, axis=1), use_container_width=True)
                    else:
                        st.info("No transactions match your search criteria")
                else:
                    st.info("No transactions match your search criteria")
            except Exception as e:
                st.error(f"Error executing search: {e}")
    
    # Display sample transactions
    st.subheader(f"Transaction List (Last {time_filter} hours)")
    transactions = get_transactions(limit=50, fraud_only=fraud_filter, time_window=time_filter)
    
    if not transactions.empty:
        # Similar to overview page, format and display data
        display_df = transactions.copy()
        if 'prediction' in display_df.columns:
            display_df['Status'] = display_df['prediction'].apply(
                lambda x: "⚠️ FRAUD" if x == 1 else "✅ VALID"
            )
        
        # Select and rename columns for display
        columns_to_display = {
            'transaction_id': 'Transaction ID',
            'trans_date_trans_time': 'Time',
            'cc_num': 'Credit Card Number',
            'amt': 'Amount',
            'merchant': 'Merchant',
            'category': 'Category',
            'Status': 'Status'
        }
        
        cols_exist = [col for col in columns_to_display.keys() if col in display_df.columns]
        if cols_exist:
            display_df = display_df[cols_exist].rename(columns={col: columns_to_display[col] for col in cols_exist if col in columns_to_display})
            
            # Highlight fraudulent rows
            def highlight_fraud(row):
                if 'Status' in row and '⚠️ FRAUD' in row['Status']:
                    return ['background-color: rgba(255, 0, 0, 0.2)'] * len(row)
                return [''] * len(row)
            
            st.dataframe(display_df.style.apply(highlight_fraud, axis=1), use_container_width=True)
        else:
            st.dataframe(display_df, use_container_width=True)
        
        # Transaction details section (select a transaction to view details)
        if 'transaction_id' in transactions.columns:
            selected_id = st.selectbox("Select a transaction ID to view details", transactions['transaction_id'].tolist())
            
            if selected_id:
                st.subheader(f"Transaction Details: {selected_id}")
                
                # Get transaction details
                transaction_detail = transactions[transactions['transaction_id'] == selected_id].iloc[0]
                
                # Display details in 2 columns
                col1, col2 = st.columns(2)
                with col1:
                    for idx, (col, val) in enumerate(transaction_detail.items()):
                        if idx % 2 == 0:  # Left column
                            st.text(f"{col}: {val}")
                
                with col2:
                    for idx, (col, val) in enumerate(transaction_detail.items()):
                        if idx % 2 == 1:  # Right column
                            st.text(f"{col}: {val}")
                
                # If fraudulent, display reason
                if 'prediction' in transaction_detail and transaction_detail['prediction'] == 1:
                    st.error("This transaction was detected as FRAUDULENT")
                    st.info("Detection reason: Identified by model " + transaction_detail.get('model', 'Unknown'))
    else:
        st.info("No transactions to display")

# DETAILED STATISTICS PAGE
elif page == "Detailed Statistics":
    st.title("Detailed Fraud Statistics")
    
    # Fraud rate over time chart
    st.subheader("Fraud Rate Over Time")
    
    time_data = get_time_series_data(time_filter)
    if not time_data.empty:
        time_data['fraud_rate'] = (time_data['fraud_count'] / time_data['total_count'] * 100).fillna(0)
        
        fig = px.line(
            time_data,
            x='hour',
            y='fraud_rate',
            title=f'Fraud Rate (Last {time_filter} hours)',
            labels={'hour': 'Time', 'fraud_rate': 'Fraud Rate (%)'}
        )
        fig.update_traces(line_color='red')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No time series data available")
    
    # Fraud distribution by different attributes
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fraud Distribution by Category")
        # Using the merchant category
        distribution = get_fraud_distribution('category')
        if not distribution.empty and 'category' in distribution.columns:
            fig = px.bar(
                distribution,
                x='category',
                y='fraud_rate',
                title='Fraud Rate by Category',
                labels={'category': 'Category', 'fraud_rate': 'Fraud Rate (%)'},
                color='fraud_rate',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No category data available")
    
    with col2:
        st.subheader("Fraud Distribution by Amount Range")
        # Using custom amount ranges
        distribution = get_amount_distribution()
        if not distribution.empty and 'amount_category' in distribution.columns:
            fig = px.bar(
                distribution,
                x='amount_category',
                y='fraud_rate',
                title='Fraud Rate by Amount Range',
                labels={'amount_category': 'Amount Range', 'fraud_rate': 'Fraud Rate (%)'},
                color='fraud_rate',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No amount range data available")
    
    # Display recent alerts
    st.subheader("Recent Fraud Alerts")
    
    session = get_cassandra_connection()
    if session:
        try:
            # Query to get recent alerts
            query = "SELECT * FROM alerts LIMIT 10"
            alerts = session.execute(query)
            alerts_df = pd.DataFrame(list(alerts))
            
            if not alerts_df.empty:
                st.dataframe(alerts_df, use_container_width=True)
            else:
                st.info("No recent alerts found")
                
        except Exception as e:
            st.error(f"Error fetching alerts: {e}")

# Add button to download report
if st.button("Download CSV Report"):
    try:
        session = get_cassandra_connection()
        if session:
            # Get data for report - using Cassandra session
            try:
                # Get transactions - with a reasonable LIMIT
                query = "SELECT * FROM transactions_pred LIMIT 5000 ALLOW FILTERING"
                rows = session.execute(query)
                
                # Convert to DataFrame
                report_data = pd.DataFrame(list(rows))
                
                if not report_data.empty:
                    # Apply time filtering in Python
                    if 'trans_date_trans_time' in report_data.columns:
                        report_data['trans_date_trans_time'] = pd.to_datetime(report_data['trans_date_trans_time'])
                        # Filter by time in Python
                        start_date = df['trans_date_trans_time'].min()  # Use your historical data baseline
                        end_date = start_date + pd.Timedelta(hours=24)
                        report_data = report_data[(report_data['trans_date_trans_time'] >= start_date) & 
                                                (report_data['trans_date_trans_time'] <= end_date)]
                    
                    # Convert to CSV
                    csv = report_data.to_csv(index=False)
                    
                    # Create download link
                    st.download_button(
                        label="Download Full Report",
                        data=csv,
                        file_name=f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No data available for report in the last 24 hours")
                    
            except Exception as e:
                st.error(f"Error fetching report data: {e}")
        else:
            st.error("Cannot connect to Cassandra database")
            
    except Exception as e:
        st.error(f"Error generating report: {e}")

# Add footer
st.markdown("---")
st.markdown("""
**Real-Time Fraud Detection System © 2025**  
Créé par :  
- Aya Chakour  
- Nabila Chaou  
- Karima Chakkour  
Étudiantes à l'ENSA Tétouan
""")

