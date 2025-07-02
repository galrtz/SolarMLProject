def export_forecast_all_nodes(model, graphs, device, output_dir):
    model.eval()

    horizons = {0: "t+15", 1: "t+30", 2: "t+45", 3: "t+60"}
    offsets = {0: 15, 1: 30, 2: 45, 3: 60}

    graph_counter = 0

    with torch.no_grad():
        for graph in graphs:
            graph_counter += 1
            graph = graph.to(device)
            output = model(graph.x, graph.edge_index, graph.edge_attr)
            pred = output.cpu().numpy()
            true = graph.y.cpu().numpy()
            x = graph.x.cpu().numpy()

            try:
                base_time = graph.date - timedelta(hours=1)
                print(f"Graph {graph_counter} | base_time: {base_time.strftime('%Y-%m-%d %H:%M')}")

                for node_idx in range(x.shape[0]):
                    plot_rows = []

                    for j in range(4):  # For each horizon
                        forecast_time = base_time + timedelta(minutes=offsets[j])
                        time_str = forecast_time.strftime("%Y-%m-%d %H:%M")
                        horizon_str = horizons[j]

                        row = {
                            "forecast_time": time_str,
                            "prediction": pred[node_idx, j],
                            "target": true[node_idx, j]
                        }

                        # Save CSV per (graph, node, horizon)
                        subdir = os.path.join(output_dir, "per_node_horizon_csvs")
                        os.makedirs(subdir, exist_ok=True)
                        filename = f"forecast_graph{graph_counter}_node{node_idx}_{horizon_str}.csv"
                        path = os.path.join(subdir, filename)

                        # Write single row (append mode)
                        df_row = pd.DataFrame([row])
                        write_header = not os.path.exists(path)
                        df_row.to_csv(path, mode='a', header=write_header, index=False)

                        # Collect for plotting (only node 0)
                        if node_idx == 0:
                            plot_rows.append({
                                "forecast_time": forecast_time,
                                "prediction": pred[node_idx, j],
                                "target": true[node_idx, j],
                                "horizon": horizon_str
                            })

                    # Plot only for node 0
                    if node_idx == 0:
                        df_plot = pd.DataFrame(plot_rows)
                        plot_path = os.path.join(output_dir, f"forecast_graph{graph_counter}_node0_plot.png")
                        plot_forecast_subplots(df_plot, plot_path, f"Forecast for Graph {graph_counter} - Node 0")
                        print(f"Saved plot for node 0: {plot_path}")

            except Exception as e:
                print(f"Error in graph {graph_counter}: {e}")
