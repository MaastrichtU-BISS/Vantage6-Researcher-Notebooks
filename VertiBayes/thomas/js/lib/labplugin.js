var plugin = require('./index');
var base = require('@jupyter-widgets/base');

module.exports = {
  id: 'thomas-jupyter-widget:plugin',
  requires: [base.IJupyterWidgetRegistry],
  activate: function(app, widgets) {
      widgets.registerWidget({
          name: 'thomas-jupyter-widget',
          version: plugin.version,
          exports: plugin
      });
  },
  autoStart: true
};

