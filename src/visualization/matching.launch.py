"""显示机场栅格地图，可选叠加已保存的抵近匹配回放。"""
from pathlib import Path
import tempfile
import yaml
import json
import math
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnShutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = next(p for p in HERE.parents if (p / 'environment.yml').is_file())

def start_rviz(context):
    path = PROJECT_ROOT / 'configs/rviz/airport.rviz'
    cleanup = []
    if LaunchConfiguration('replay').perform(context).lower() in ('true', '1'):
        config = yaml.safe_load(path.read_text())
        manager = config['Visualization Manager']
        for display in manager['Displays']:
            if display['Name'] in ('Plan Markers', 'Drone Traj'):
                display['Enabled'] = display['Value'] = True
        manager['Displays'].append({
            'Class': 'rviz_default_plugins/Image', 'Name': 'Control Weights and Matching Point',
            'Enabled': True, 'Value': True,
            'Topic': {'Value': '/near_nose/plan/view', 'Depth': 5,
                      'Durability Policy': 'Volatile', 'Reliability Policy': 'Reliable',
                      'History Policy': 'Keep Last'}})
        manager['Views']['Current']['Distance'] = 70
        manager['Views']['Current']['Focal Point']['Z'] = 15
        setup_path=Path(LaunchConfiguration('dump_dir').perform(context)).parent/'controller-setup.json'
        if setup_path.exists():
            setup=json.loads(setup_path.read_text())
            if setup.get('mode')=='oblique' and 'start_ned_m' in setup:
                a,b=setup['start_ned_m'],setup['reference_ned_m']
                manager['Views']['Current']['Distance']=max(130.,1.8*math.sqrt(sum((x-y)**2 for x,y in zip(a,b))))
                manager['Views']['Current']['Focal Point']={'X':(a[1]+b[1])/2,'Y':(a[0]+b[0])/2,'Z':-(a[2]+b[2])/2}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rviz', prefix='near-nose-live-', delete=False) as f:
            yaml.safe_dump(config, f, sort_keys=False)
            path = Path(f.name)
        def remove_config(context):
            path.unlink(missing_ok=True)
            return []
        cleanup.append(RegisterEventHandler(OnShutdown(on_shutdown=[OpaqueFunction(function=remove_config)])))
    return cleanup + [ExecuteProcess(cmd=['rviz2', '-d', str(path)], output='screen')]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('profile', default_value='airport'),
        DeclareLaunchArgument('rate', default_value='5.0'),
        DeclareLaunchArgument('heatmap_mode', default_value='control'),
        DeclareLaunchArgument('replay', default_value='false'),
        DeclareLaunchArgument('dump_dir', default_value=str(PROJECT_ROOT / 'runs/approach/close4-1793954736186/viz_dump')),
        DeclareLaunchArgument('start_rviz', default_value='true'),
        DeclareLaunchArgument('publish_map', default_value='true'),
        DeclareLaunchArgument('map_dir', default_value=''),
        ExecuteProcess(
            cmd=['/usr/bin/python3', str(HERE / 'map_view.py'),
                 '--profile', LaunchConfiguration('profile'),
                 '--map-dir', LaunchConfiguration('map_dir')],
            output='screen', condition=IfCondition(LaunchConfiguration('publish_map'))),
        ExecuteProcess(
            cmd=['/usr/bin/python3', str(PROJECT_ROOT / 'src/visualization/plan_viz_node.py'),
                 '--dump-dir', LaunchConfiguration('dump_dir'),
                 '--node-name', 'near_nose_replay',
                 '--topic-prefix', '/near_nose',
                 '--heatmap-mode', LaunchConfiguration('heatmap_mode'),
                 '--softmax-temperature', '0.005',
                 '--rate', LaunchConfiguration('rate')],
            output='screen', condition=IfCondition(LaunchConfiguration('replay'))),
        OpaqueFunction(function=start_rviz, condition=IfCondition(LaunchConfiguration('start_rviz'))),
    ])
