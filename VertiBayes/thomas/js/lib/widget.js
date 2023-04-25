var widgets = require('@jupyter-widgets/base');
var _ = require('lodash');

var Konva = require('konva');
const { v1: uuid1 } = require('uuid');

// See widget.py for the kernel counterpart to this file.


// Custom Model. Custom widgets models must at least provide default values
// for model attributes, including
//
//  - `_view_name`
//  - `_view_module`
//  - `_view_module_version`
//
//  - `_model_name`
//  - `_model_module`
//  - `_model_module_version`
//
//  when different from the base class.

// When serializing the entire widget state for embedding, only values that
// differ from the defaults will be specified.
var Model = widgets.DOMWidgetModel.extend({
    defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
        _view_name : 'View',
        _model_name : 'Model',
        _view_module : 'thomas-jupyter-widget',
        _model_module : 'thomas-jupyter-widget',
        _model_module_version : '0.1.0',
        _view_module_version : '0.1.0',
        value : {},
        marginals_and_evidence : {},
        evidence_sink: '',
        height: 300,
    })
});


function intersect(x1, y1, x2, y2, x3, y3, x4, y4) {

    // Check if none of the lines are of length 0
    if ((x1 === x2 && y1 === y2) || (x3 === x4 && y3 === y4)) {
        console.warn('Found line of length 0');
        return false
    }

    var denominator = ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))

    // Lines are parallel
    if (denominator === 0) {
        console.warn('Denominator is zero 0');
        return false
    }

    let ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
    let ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator

    // is the intersection along the segments
    if (ua < 0 || ua > 1 || ub < 0 || ub > 1) {
        // console.debug('Intersection outside segments');
        return false
    }

    // Return an object with the x and y coordinates of the intersection
    let x = x1 + ua * (x2 - x1)
    let y = y1 + ua * (y2 - y1)

    return {x, y}
}

function compute_intersection(corners, line) {
    const points = [
        ['tl', 'tr'],
        ['tl', 'bl'],
        ['tr', 'br'],
        ['bl', 'br'],
    ];

    var intersection = false;

    for (const p of points) {
        const key1 = p[0], key2 = p[1];

        intersection = intersect(
            corners[key1].x,
            corners[key1].y,
            corners[key2].x,
            corners[key2].y,
            line.src.x,
            line.src.y,
            line.dst.x,
            line.dst.y,
        );

        if (intersection) {
            break;
        }
    }

    if (!intersection) {
        console.warn('Could not determine intersection');
    }

    return intersection;
}


class Node extends Konva.Group {

    /**
     * Create a new edge between two nodes.
     *
     * Args:
     *   node (object): part of the JSON that defines a Node.
     *   marginals (dict): dict of marginals, indexed by state
     */
    constructor(node, marginals, evidence, eventhandlers) {
        // First things first
        super({
            x: node.position[0],
            y: node.position[1],
            draggable: true,
        });

        const {
            onDragMove,
            onDragEnd,
            onStateSelected,
        } = eventhandlers;

        this.RV = node.RV;
        this.name = node.name;
        this.node = node;
        this.edges = [];

        this._title_height = 15;
        this._state_offset = 8;
        this._state_height = 14;
        this._state_padding = 2;

        this._width = 180
        this._height = this.computeHeight();
        this.width(this._width);
        this.height(this._height);

        this.createBackground();
        this.createTitle();
        this.createStates(marginals, evidence, onStateSelected);


        this.on('dragmove', () => onDragMove(this));
        this.on('dragend', () => onDragEnd(this));
    }

    addEdge(edge) {
        this.edges.push(edge);
    }

    getCenter() {
        return {
            x: this.x() + this.width() / 2,
            y: this.y() + this.height() / 2
        }
    }

    getCorners() {
        return {
            'tl': {x: this.x(), y: this.y()},
            'tr': {x: this.x() + this.width(), y: this.y()},
            'bl': {x: this.x(), y: this.y() + this.height()},
            'br': {x: this.x() + this.width(), y: this.y() + this.height()},
        }
    }

    /**
     * Compute the node's height based on the number of states.
     */
    computeHeight() {
        const { node } = this;
        return (
            node.states.length * this._state_height
            + 2 * this._state_offset
            + this._title_height
        )
    }

    /**
     * Create an opaque background.
     */
    createBackground() {
        this.add(
            new Konva.Rect({
                fill: '#efefef',
                width: this._width,
                height: this._height,
                cornerRadius: 5,
                shadowBlur: 5,
            })
        );
    }

    /**
     * Create a label to display the RV.
     */
    createTitle() {
        const { node } = this;

        // Node's RV in the top-left
        const label = new Konva.Label();
        label.add(
            new Konva.Text({
                text: node.RV,
                padding: 4,
                fontSize: this._title_height,
                fontStyle: "bold",
                width: this._width,
            })
        );

        // If full name differs from RV, display it behind the RV
        if (node.RV != node.name) {
            label.add(
                new Konva.Text({
                    x: 20,
                    text: `| ${node.name}`,
                    padding: 4,
                    fontSize: this._title_height - 2,
                    fontStyle: "normal",
                    verticalAlign: "bottom",
                    width: this._width,
                })
            );
        }

        this.add(label);
    }

    /**
     * Create a Group to hold a state's shapes.
     *
     * Args:
     *   state (str): name/identifier of the state.
     *   idx (int): position in the list of states
     *   marginals (dict): dict of marginals, indexed by state
     */
    createState(state, idx, marginals, evidence, onStateSelected) {
        const y = (
            this._title_height
            + this._state_offset
            + idx * this._state_height
        )

        const
            label_width = 70,
            marginal_width = 55;

        const remaining_width = this._width - label_width - marginal_width;

        var marginal = '...';
        var bar_width = 0;
        var bar_color = '#003366';

        if (evidence && evidence === state) {
            bar_color = '#00BCCC';
        }

        if (marginals) {
            marginal = (100 * marginals[state]).toFixed(2) + '%';
            bar_width = 1 + remaining_width * marginals[state]
        }

        // Create the Group
        const group = new Konva.Group({y: y});

        // State label
        group.add(
            new Konva.Label().add(
                new Konva.Text({
                    text: state,
                    padding: this._state_padding,
                    fontSize: this._state_height - this._state_padding,
                    wrap: 'none',
                    ellipsis: 'ellipsis',
                    width: this._width,
                })
            )
        );

        // State bar
        group.add(
            new Konva.Rect({
                x: label_width,
                y: 1,
                width: bar_width,
                height: this._state_height - 2,
                fill: bar_color,
            })
        );

        // State marginal
        group.add(
            new Konva.Label({
                x: this._width - marginal_width
            }).add(
                new Konva.Text({
                    text: marginal,
                    padding: this._state_padding,
                    fontSize: this._state_height - this._state_padding,
                    align: "right",
                    wrap: "none",
                    width: marginal_width,
                })
            )
        );

        group.on('dblclick', () => onStateSelected(this.RV, state))
        this.add(group);
    }

    createStates(marginals, evidence, onStateSelected) {
        const { states } = this.node;
        states.map((state, idx) => {
            this.createState(state, idx, marginals, evidence, onStateSelected)
        });
    }
}

class Edge extends Konva.Arrow {
    /**
     * Create a new edge between two nodes.
     *
     * Args:
     *   src (Node): src
     *   dst (Node): dst
     */
    constructor(src, dst) {
        super({
            x: 0,
            y: 0,
            // points: [src_i.x, src_i.y, dst_i.x, dst_i.y],
            pointerLength: 10,
            pointerWidth: 10,
            fill: 'black',
            stroke: 'black',
            strokeWidth: 2,
        });

        this.src = src;
        this.dst = dst;

        this.recomputePoints();

        src.addEdge(this);
        dst.addEdge(this);
    }

    /**
     * Recompute the arrow src -> dst.
     */
    recomputePoints() {
        const { src, dst } = this;

        const src_center = src.getCenter();
        const dst_center = dst.getCenter();

        const src_i = compute_intersection(
            src.getCorners(),
            {src: src_center, dst: dst_center}
        );

        const dst_i = compute_intersection(
            dst.getCorners(),
            {src: src_center, dst: dst_center}
        );

        this.points([src_i.x, src_i.y, dst_i.x, dst_i.y]);
    }

    /**
     * Called by View when a Node moved.
     */
    onNodeMoving() {
        this.recomputePoints();
    }
}


// Custom View. Renders the widget model.
var View = widgets.DOMWidgetView.extend({

    // Defines how the widget gets rendered into the DOM
    render: function() {
        // this.model refers to the *Python* model associated with this widget.
        var height = this.model.get('height');
        // console.log("And everyday I'm rendering", height);

        this.container_id = `konva-container-${uuid1()}`
        this.node_title_height = 15;
        this.node_state_offset = 8;
        this.node_state_height = 14;
        this.node_state_padding = 2;

        this.node_width = 180;
        this.nodes = [];
        this.edges = [];
        this.map = {};

        this.el.innerHTML = `
            <div>
                <!--
                <div style="padding: 10px; background-color: #336699">
                    <button id="save">Save as image</button>
                </div>
                -->
                <div
                    id="${this.container_id}"
                    style="background-color: #336699"
                    >
                </div>
            </div>
        `;

        // Run this *after* the above <div> has rendered.
        setTimeout(() => {
            // console.log('Setting up Konva ...');
            this.stage = new Konva.Stage({
              container: this.container_id,
              width: 2048,
              height: height
            });

            // Create a Layer to hold all shapes
            this.layer = new Konva.Layer();

            this.model.on('change:value', this.value_changed, this);
            this.model.on('change:marginals_and_evidence', this.value_changed, this);

            /*
            document.getElementById('save').addEventListener(
                'click',
                () => {
                  // var dataURL = this.layer.toDataURL({ pixelRatio: 3 });
                  var dataURL = this.layer.toDataURL();
                  this.downloadURI(dataURL, 'stage.png');
                },
                false
            );
            */

            this.value_changed();
        }, 0)
    },

    downloadURI: function(uri, name) {
        var link = document.createElement('a');
        link.download = name;
        link.href = uri;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        delete link;
    },

    /**
     * Called once when this.model.get('value') changes.
     * This should trigger a complete re-render of the canvas.
     */
    value_changed: function() {
        // console.log('value_changed()');
        // value holds the output of BayesianNetwork.as_dict()
        var value = this.model.get('value');
        var { marginals, evidence } = this.model.get('marginals_and_evidence');

        if (value.type !== 'BayesianNetwork') {
            return
        }

        // console.log('marginals:', marginals);
        // console.log('evidence:', evidence);

        // Clear the layer
        this.layer.removeChildren();

        // Create nodes & mapping (indexed by RV)
        this.map = {};
        var n;

        this.nodes = value.nodes.map(node => {
            n = new Node(
                node,
                marginals[node.RV],
                evidence[node.RV], {
                    onDragMove: (n) => this.on_node_moving(n),
                    onDragEnd: (n) => this.on_node_moved(n),
                    onStateSelected: (RV, state) => this.on_state_selected(RV, state),
                }
            );

            this.map[node.RV] = n;
            return n;
        });

        // Create edges
        this.edges = value.edges.map(e => {
            const
                src = this.map[e[0]],
                dst = this.map[e[1]];

            return new Edge(src, dst);
        })

        // Add nodes & edges to the layer to the stage.
        this.edges.forEach(i => this.layer.add(i));
        this.nodes.forEach(i => this.layer.add(i));
        this.stage.add(this.layer);

        this.layer.draw();
    },

    on_node_moving(node) {
        node.edges.forEach(e => e.onNodeMoving());
        this.layer.draw();
    },

    on_node_moved(node) {
        console.log(`node ${node.RV} moved!`);

        // node.node contains a reference to the node's JSON definition
        node.node.position = [node.x(), node.y()];

        var value = this.model.get('value');
        // For some reason it is necessary to set value twice.
        this.model.set('value', 'null');
        this.model.set('value', Object.assign({}, value));
        this.touch();
    },

    on_state_selected(RV, state) {
        // console.log(`on_state_selected("${RV}", "${state}")`);

        const { marginals, evidence } = this.model.get('marginals_and_evidence');
        // console.log('evidence: ', evidence);

        const e = Object.assign({}, evidence);

        if (e[RV] && e[RV] === state) {
            // console.log('  disabling state ...')
            e[RV] = ''
        } else {
            // console.log('  setting evidence ...')
            e[RV] = state;
        }

        // console.log('e: ', e);
        this.model.set('evidence_sink', e);
        this.touch();
    }
});


module.exports = {
    Model: Model,
    View: View
};
