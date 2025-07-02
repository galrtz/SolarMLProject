def export_forecast_all_nodes(model, graphs, device, output_dir):
    model.eval()

    results = {0: [], 1: [], 2: [], 3: []}
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

                matches = range(x.shape[0])  # All nodes in the graph

                for i in matches:
                    for j in range(4):  # 4 forecast horizons: t+15 .. t+60
                        forecast_time = base_time + timedelta(minutes=offsets[j])
                        time_str = forecast_time.strftime("%Y-%m-%d %H:%M")

                        results[j].append({
                            "forecast_time": time_str,
                            "prediction": pred[i, j],
                            "target": true[i, j],
                            "node_idx": i,
                            "graph_idx": graph_counter,
                            "lon": x[i, 2],
                            "lat": x[i, 3]
                        })

            except Exception as e:
                print(f"Error in graph {graph_counter}: {e}")

    all_results = []
    for j in range(4):
        if not results[j]:
            print(f"No data found for horizon {horizons[j]}")
            continue
        df = pd.DataFrame(results[j])
        df = df.sort_values("forecast_time")
        label = horizons[j]

        for row in results[j]:
            row["horizon"] = horizons[j]
            all_results.append(row)

        csv_filename = os.path.join(output_dir, f"all_nodes_forecast_{label}.csv")
        df.to_csv(csv_filename, index=False)
        print(f"Saved CSV: {csv_filename}")

    # Plotting by date
    df_all = pd.DataFrame(all_results)
    df_all["forecast_time"] = pd.to_datetime(df_all["forecast_time"], errors="coerce")
    df_all = df_all.dropna(subset=["forecast_time"])
    df_all["date"] = df_all["forecast_time"].dt.date

    unique_dates = df_all["date"].unique()

    for date in sorted(unique_dates):
        df_day = df_all[df_all["date"] == date].copy()
        if df_day.empty:
            continue
        date_str = date.strftime("%Y-%m-%d")
        png_path = os.path.join(output_dir, f"all_nodes_combined_forecast_{date_str}.png")
        plot_forecast_subplots(df_day, png_path, f"Forecast for All Nodes - {date_str}")
        print(f"Saved Plot: {png_path}")
