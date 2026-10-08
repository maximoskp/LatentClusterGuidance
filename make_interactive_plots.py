import pickle
import plotly.graph_objects as go
import os
import numpy as np
from GridMLM_tokenizers import CSGridMLMTokenizer
import torch

tokenizer = CSGridMLMTokenizer(
    fixed_length=80,
    quantization='4th',
    intertwine_bar_info=True,
    trim_start=False,
    use_pc_roll=True,
    use_full_range_melody=False
)

# auxilliary functions
def flatten_piece_metadata(latent_steps_list, key):
    return torch.cat([
        item[key].reshape(-1)
        for item in latent_steps_list
    ]).cpu().numpy()
# end flatten_piece_metadata

def make_interactive_plot_for_layer(mod_name, k):
    fig = go.Figure()

    groups = [
        (
            "GJT train",
            umap_2d_per_layer[k]["gjt_train_2d"],
            gjt_train_harmony_ids,
            gjt_train_piece_ids,
            gjt_train_step_indices,
            "firebrick",
        ),
        (
            "GJT validation",
            umap_2d_per_layer[k]["gjt_val_2d"],
            gjt_val_harmony_ids,
            gjt_val_piece_ids,
            gjt_val_step_indices,
            "fuchsia",
        ),
        (
            "NoTT train",
            umap_2d_per_layer[k]["nott_train_2d"],
            nott_train_harmony_ids,
            nott_train_piece_ids,
            nott_train_step_indices,
            "blue",
        ),
        (
            "NoTT validation",
            umap_2d_per_layer[k]["nott_val_2d"],
            nott_val_harmony_ids,
            nott_val_piece_ids,
            nott_val_step_indices,
            "cyan",
        ),
    ]

    for name, coords, harmony_ids, piece_ids, step_indices, color in groups:
        harmony_ids = np.asarray(harmony_ids).reshape(-1)
        piece_ids = np.asarray(piece_ids).reshape(-1)
        step_indices = np.asarray(step_indices).reshape(-1)

        harmony_tokens = np.asarray([
            tokenizer.ids_to_tokens[int(token_id)]
            for token_id in harmony_ids
        ])

        assert len(coords) == len(harmony_ids) == len(piece_ids) == len(step_indices), (
            f"{name}: coordinates and point metadata differ in length"
        )

        # Columns are harmony token, piece ID, and position within the piece.
        customdata = np.column_stack((harmony_tokens, piece_ids, step_indices))

        fig.add_trace(go.Scattergl(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            name=name,
            customdata=customdata,
            marker=dict(color=color, size=5, opacity=0.35),
            hovertemplate=(
                f"{name}<br>"
                "Piece ID: %{customdata[1]}<br>"
                "Harmony: %{customdata[0]}<br>"
                "Position: %{customdata[2]}<br>"
                "UMAP 1: %{x:.3f}<br>"
                "UMAP 2: %{y:.3f}"
                "<extra></extra>"
            ),
        ))

    # This trace is updated by the hover handler; it does not affect marker colors.
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode="lines",
        line=dict(width=2),
        hoverinfo="skip",
        showlegend=False,
    ))
    overlay_index = len(groups)

    post_script = """
    const plot = document.getElementById('{plot_id}');
    const overlayIndex = __OVERLAY_INDEX__;
    const stepCount = 80;
    const sourceTraces = plot.data.slice(0, overlayIndex).map(trace => ({
        x: Array.from(trace.x),
        y: Array.from(trace.y),
        customdata: trace.customdata.map(data => Array.from(data))
    }));

    const controls = document.createElement('div');
    controls.style.cssText = [
        'position:fixed',
        'top:12px',
        'right:12px',
        'z-index:1000',
        'padding:10px',
        'background:rgba(255,255,255,0.92)',
        'border:1px solid #aaa',
        'border-radius:4px',
        'max-height:80vh',
        'overflow-y:auto',
        'font:12px sans-serif'
    ].join(';');

    const controlsTitle = document.createElement('div');
    controlsTitle.textContent = 'Step indices';
    controlsTitle.style.cssText = 'font-weight:bold;margin-bottom:6px';
    controls.appendChild(controlsTitle);

    const checkboxGrid = document.createElement('div');
    checkboxGrid.style.cssText = 'display:grid;grid-template-columns:repeat(8,auto);gap:4px 8px';
    for (let step = 0; step < stepCount; step++) {
        const label = document.createElement('label');
        label.style.cssText = 'display:flex;align-items:center;gap:2px';
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = true;
        checkbox.dataset.step = String(step);
        checkbox.addEventListener('change', updateVisibleSteps);
        label.append(checkbox, document.createTextNode(String(step)));
        checkboxGrid.appendChild(label);
    }
    controls.appendChild(checkboxGrid);
    document.body.appendChild(controls);

    const tooltipStyle = document.createElement('style');
    tooltipStyle.textContent =
        '.hoverlayer .hovertext path, .hoverlayer .hovertext rect {' +
        'fill-opacity:0.5 !important;}';
    document.head.appendChild(tooltipStyle);

    function updateVisibleSteps() {
        const selectedSteps = new Set(
            Array.from(controls.querySelectorAll('input[data-step]:checked'))
                .map(checkbox => Number(checkbox.dataset.step))
        );
        sourceTraces.forEach((trace, traceIndex) => {
            const visibleIndices = trace.customdata.reduce((indices, data, index) => {
                if (selectedSteps.has(Number(data[2]))) indices.push(index);
                return indices;
            }, []);
            Plotly.restyle(plot, {
                x: [visibleIndices.map(index => trace.x[index])],
                y: [visibleIndices.map(index => trace.y[index])],
                customdata: [visibleIndices.map(index => trace.customdata[index])]
            }, [traceIndex]);
        });
    }

    plot.on('plotly_hover', (event) => {
        const point = event.points[0];
        if (point.curveNumber >= overlayIndex) return;

        const trace = plot._fullData[point.curveNumber];
        const pieceId = String(point.customdata[1]);

        const pointsInPiece = trace.customdata
            .map((data, i) => ({
                index: i,
                pieceId: String(data[1]),
                step: Number(data[2])
            }))
            .filter(item => item.pieceId === pieceId)
            .sort((a, b) => a.step - b.step);

        console.log({
            trace: point.curveNumber,
            pieceId,
            matchingPoints: pointsInPiece.length,
            firstSteps: pointsInPiece.slice(0, 5).map(item => item.step)
        });

        const x = pointsInPiece.map(item => trace.x[item.index]);
        const y = pointsInPiece.map(item => trace.y[item.index]);

        Plotly.restyle(plot, {
            x: [x],
            y: [y],
            'line.color': [trace.marker.color],
            'line.width': [2]
        }, [overlayIndex])
    });

    plot.on('plotly_unhover', () => {
        Plotly.restyle(plot, {x: [[]], y: [[]]}, [overlayIndex]);
    });
    """.replace("__OVERLAY_INDEX__", str(overlay_index))

    fig.update_layout(
        title=f"UMAP of latent layer {k}",
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        hovermode="closest",
        hoverlabel=dict(bgcolor="rgba(255,255,255,0.5)"),
    )

    os.makedirs("html/", exist_ok=True)
    os.makedirs(f"html/{mod_name}", exist_ok=True)
    fig.write_html(
        f"html/{mod_name}/umap_{k}.html",
        include_plotlyjs="cdn",
        post_script=post_script,
    )
make_interactive_plot_for_layer

# Define models ===============================================
models_data = []
models_data.append({
    'path': 'saved_models/SELaC_pretrained/pretrained_epoch89_nvis0.pt',
    'name': 'epoch89_nvis0'
})
models_data.append({
    'path': 'saved_models/SELaC_pretrained/pretrained_epoch130_nvis50.pt',
    'name': 'epoch130_nvis50'
})
models_data.append({
    'path': 'saved_models/SELaC_pretrained/pretrained_epoch168_nvis50.pt',
    'name': 'epoch168_nvis50'
})
models_data.append({
    'path': 'saved_models/SELaC_pretrained/pretrained_epoch200_nvis3.pt',
    'name': 'epoch200_nvis3'
})

for model_data in models_data:
    model_name = model_data['name']
    print(f'running for model: {model_name}')

    # loading harmony ids
    print('loading harmony ids')
    with open(f'data/{model_name}/gjt_train_harmony_ids.pickle', 'rb') as handle:
        gjt_train_harmony_ids = pickle.load(handle)
    with open(f'data/{model_name}/gjt_val_harmony_ids.pickle', 'rb') as handle:
        gjt_val_harmony_ids = pickle.load(handle)

    with open(f'data/{model_name}/nott_train_harmony_ids.pickle', 'rb') as handle:
        nott_train_harmony_ids = pickle.load(handle)
    with open(f'data/{model_name}/nott_val_harmony_ids.pickle', 'rb') as handle:
        nott_val_harmony_ids = pickle.load(handle)

    # loading steps dict
    print('loading steps dict')
    with open(f'data/{model_name}/gjt_train_latent_steps_dict.pickle', 'rb') as handle:
        gjt_train_latent_steps_dict = pickle.load(handle)
    with open(f'data/{model_name}/gjt_val_latent_steps_dict.pickle', 'rb') as handle:
        gjt_val_latent_steps_dict = pickle.load(handle)

    with open(f'data/{model_name}/nott_train_latent_steps_dict.pickle', 'rb') as handle:
        nott_train_latent_steps_dict = pickle.load(handle)
    with open(f'data/{model_name}/nott_val_latent_steps_dict.pickle', 'rb') as handle:
        nott_val_latent_steps_dict = pickle.load(handle)

    # loading steps list
    print('loading steps list')
    with open(f'data/{model_name}/gjt_train_latent_steps_list.pickle', 'rb') as handle:
        gjt_train_latent_steps_list = pickle.load(handle)
    with open(f'data/{model_name}/gjt_val_latent_steps_list.pickle', 'rb') as handle:
        gjt_val_latent_steps_list = pickle.load(handle)

    with open(f'data/{model_name}/nott_train_latent_steps_list.pickle', 'rb') as handle:
        nott_train_latent_steps_list = pickle.load(handle)
    with open(f'data/{model_name}/nott_val_latent_steps_list.pickle', 'rb') as handle:
        nott_val_latent_steps_list = pickle.load(handle)

    # loading umap
    print('loading umap')
    with open(f'data/{model_name}/umap/umap_2d_per_layer.pickle', 'rb') as handle:
        umap_2d_per_layer = pickle.load(handle)

    print('flattening metadata')
    gjt_train_piece_ids = flatten_piece_metadata(gjt_train_latent_steps_list, "piece_ids")
    gjt_train_step_indices = flatten_piece_metadata(gjt_train_latent_steps_list, "step_indices")

    gjt_val_piece_ids = flatten_piece_metadata(gjt_val_latent_steps_list, "piece_ids")
    gjt_val_step_indices = flatten_piece_metadata(gjt_val_latent_steps_list, "step_indices")

    nott_train_piece_ids = flatten_piece_metadata(nott_train_latent_steps_list, "piece_ids")
    nott_train_step_indices = flatten_piece_metadata(nott_train_latent_steps_list, "step_indices")

    nott_val_piece_ids = flatten_piece_metadata(nott_val_latent_steps_list, "piece_ids")
    nott_val_step_indices = flatten_piece_metadata(nott_val_latent_steps_list, "step_indices")

    for k in range(8):
        print(f'making interactive plot for model {model_name} - layer {k}')
        make_interactive_plot_for_layer(model_name, k)