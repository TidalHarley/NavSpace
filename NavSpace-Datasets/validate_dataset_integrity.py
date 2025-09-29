import json
import os
from typing import Dict, List, Any, Tuple

def load_json_file(file_path: str) -> Any:
    """安全加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ 文件不存在: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ JSON格式错误 {file_path}: {e}")
        return None
    except Exception as e:
        print(f"❌ 加载文件 {file_path} 时出错: {e}")
        return None

def extract_episode_data(data: Any, file_type: str) -> Dict[str, Dict]:
    """从JSON数据中提取episode信息"""
    episodes = {}
    if data is None or 'episodes' not in data:
        return episodes
    
    for episode in data['episodes']:
        episode_id = episode.get('episode_id')
        if episode_id is None:
            continue
            
        episode_data = {
            'episode_id': episode_id,
            'trajectory_id': episode.get('trajectory_id'),
            'scene_id': episode.get('scene_id'),
            'instruction_text': None,
            'start_position': episode.get('start_position'),
            'start_rotation': episode.get('start_rotation'),
            'goals': episode.get('goals'),
            'info': episode.get('info'),
            'file_type': file_type
        }
        
        # 处理instruction字段
        if 'instruction' in episode:
            if isinstance(episode['instruction'], dict) and 'instruction_text' in episode['instruction']:
                episode_data['instruction_text'] = episode['instruction']['instruction_text']
            elif isinstance(episode['instruction'], str):
                episode_data['instruction_text'] = episode['instruction']
        
        # 处理action.json文件的特殊字段
        if file_type == 'action':
            episode_data['actions'] = episode.get('actions')
            episode_data['reference_path'] = episode.get('reference_path')
        
        # 处理with_tokens.json文件的特殊字段
        if file_type == 'tokens':
            if 'instruction' in episode:
                episode_data['instruction_tokens'] = episode['instruction'].get('tokens')
                episode_data['instruction_token_ids'] = episode['instruction'].get('token_ids')
        
        episodes[episode_id] = episode_data
    
    return episodes

def compare_positions(pos1: List[float], pos2: List[float], tolerance: float = 1e-6) -> bool:
    """比较两个位置坐标是否相等"""
    if pos1 is None or pos2 is None:
        return pos1 == pos2
    if len(pos1) != len(pos2):
        return False
    return all(abs(a - b) < tolerance for a, b in zip(pos1, pos2))

def compare_goals(goals1: List[Dict], goals2: List[Dict], tolerance: float = 1e-6) -> bool:
    """比较两个goals列表是否相等"""
    if goals1 is None or goals2 is None:
        return goals1 == goals2
    if len(goals1) != len(goals2):
        return False
    
    for g1, g2 in zip(goals1, goals2):
        if not compare_positions(g1.get('position'), g2.get('position'), tolerance):
            return False
        if abs(g1.get('radius', 0) - g2.get('radius', 0)) >= tolerance:
            return False
    
    return True

def compare_scene_filename(scene_id1: str, scene_id2: str) -> bool:
    """比较scene_id的文件名部分（忽略路径差异）"""
    if scene_id1 is None or scene_id2 is None:
        return scene_id1 == scene_id2
    
    # 提取文件名部分
    filename1 = scene_id1.split('/')[-1] if '/' in scene_id1 else scene_id1
    filename2 = scene_id2.split('/')[-1] if '/' in scene_id2 else scene_id2
    
    return filename1 == filename2

def compare_episodes(episode1: Dict, episode2: Dict) -> Tuple[bool, List[str]]:
    """详细比较两个episode，返回是否一致和差异列表"""
    differences = []
    
    # 比较基本字段
    basic_fields = ['episode_id', 'trajectory_id']
    for field in basic_fields:
        if episode1.get(field) != episode2.get(field):
            differences.append(f"{field}不同: {episode1.get(field)} vs {episode2.get(field)}")
    
    # 比较scene_id（只比较文件名部分）
    if not compare_scene_filename(episode1.get('scene_id'), episode2.get('scene_id')):
        differences.append(f"scene_id文件名不同: {episode1.get('scene_id')} vs {episode2.get('scene_id')}")
    
    # 比较instruction
    if episode1.get('instruction_text') != episode2.get('instruction_text'):
        differences.append(f"instruction_text不同")
    
    # 比较start_position
    if not compare_positions(episode1.get('start_position'), episode2.get('start_position')):
        differences.append(f"start_position不同")
    
    # 比较start_rotation
    if not compare_positions(episode1.get('start_rotation'), episode2.get('start_rotation')):
        differences.append(f"start_rotation不同")
    
    # 比较goals
    if not compare_goals(episode1.get('goals'), episode2.get('goals')):
        differences.append(f"goals不同")
    
    # 比较info字段
    info1 = episode1.get('info', {})
    info2 = episode2.get('info', {})
    if info1.get('geodesic_distance') != info2.get('geodesic_distance'):
        differences.append(f"geodesic_distance不同")
    
    return len(differences) == 0, differences

def validate_folder_integrity(folder_name: str) -> Dict[str, Any]:
    """验证单个文件夹的数据完整性"""
    result = {
        'folder_name': folder_name,
        'files_found': {},
        'episode_counts': {},
        'all_consistent': False,
        'consistency_details': {},
        'missing_episodes': {},
        'extra_episodes': {},
        'differences': []
    }
    
    # 文件路径映射 - 根据实际文件名格式
    folder_prefix_map = {
        'Environment State': 'envstate',
        'Precise Movement': 'precisemove',
        'Space Structure': 'spacestructure',
        'Spatial Relationship': 'spatialrel',
        'Vertical Perception': 'verticalpercep',
        'Viewpoint Shifting': 'viewpointsft'
    }
    
    prefix = folder_prefix_map.get(folder_name, folder_name.replace(' ', '').lower())
    file_types = {
        'vln': f"{folder_name}/{prefix}_vln.json",
        'action': f"{folder_name}/{prefix}_action.json", 
        'tokens': f"{folder_name}/{prefix}_with_tokens.json"
    }
    
    print(f"\n{'='*80}")
    print(f"🔍 正在验证文件夹: {folder_name}")
    print(f"{'='*80}")
    
    # 加载所有文件
    all_data = {}
    for file_type, file_path in file_types.items():
        if os.path.exists(file_path):
            print(f"✅ 找到文件: {file_path}")
            data = load_json_file(file_path)
            if data is not None:
                all_data[file_type] = extract_episode_data(data, file_type)
                result['files_found'][file_type] = file_path
                result['episode_counts'][file_type] = len(all_data[file_type])
            else:
                print(f"❌ 无法加载文件: {file_path}")
                result['episode_counts'][file_type] = 0
        else:
            print(f"❌ 文件不存在: {file_path}")
            result['episode_counts'][file_type] = 0
    
    if 'vln' not in all_data:
        print("❌ 无法找到基准vln.json文件，无法进行验证")
        return result
    
    vln_episodes = all_data['vln']
    vln_ids = set(vln_episodes.keys())
    
    print(f"\n📊 Episode数量统计:")
    for file_type, count in result['episode_counts'].items():
        print(f"  {file_type:8}: {count}")
    
    # 与vln.json比较每个文件
    for file_type in ['action', 'tokens']:
        if file_type not in all_data:
            continue
            
        other_episodes = all_data[file_type]
        other_ids = set(other_episodes.keys())
        
        print(f"\n🔄 比较 vln.json 与 {file_type}.json:")
        
        # 检查episode ID对应关系
        missing_in_other = vln_ids - other_ids
        extra_in_other = other_ids - vln_ids
        common_ids = vln_ids.intersection(other_ids)
        
        if missing_in_other:
            print(f"  ⚠️  {file_type}.json中缺失的episode: {sorted(missing_in_other)}")
            result['missing_episodes'][file_type] = sorted(missing_in_other)
        
        if extra_in_other:
            print(f"  ⚠️  {file_type}.json中多余的episode: {sorted(extra_in_other)}")
            result['extra_episodes'][file_type] = sorted(extra_in_other)
        
        print(f"  📈 共同episode数量: {len(common_ids)}")
        
        # 详细比较共同的episodes
        consistent_count = 0
        inconsistent_episodes = []
        
        for episode_id in sorted(common_ids):
            vln_ep = vln_episodes[episode_id]
            other_ep = other_episodes[episode_id]
            
            is_consistent, differences = compare_episodes(vln_ep, other_ep)
            
            if is_consistent:
                consistent_count += 1
            else:
                inconsistent_episodes.append({
                    'episode_id': episode_id,
                    'differences': differences,
                    'vln_data': vln_ep,
                    'other_data': other_ep
                })
        
        consistency_rate = consistent_count / len(common_ids) * 100 if common_ids else 0
        print(f"  ✅ 完全一致的episode: {consistent_count}/{len(common_ids)} ({consistency_rate:.1f}%)")
        
        result['consistency_details'][file_type] = {
            'total_common': len(common_ids),
            'consistent': consistent_count,
            'inconsistent': len(inconsistent_episodes),
            'consistency_rate': consistency_rate
        }
        
        # 显示不一致的示例
        if inconsistent_episodes:
            print(f"  ❌ 发现 {len(inconsistent_episodes)} 个不一致的episode")
            for i, example in enumerate(inconsistent_episodes[:3]):  # 最多显示3个例子
                print(f"\n    示例 {i+1} (Episode {example['episode_id']}):")
                for diff in example['differences'][:5]:  # 最多显示5个差异
                    print(f"      - {diff}")
                if len(example['differences']) > 5:
                    print(f"      - ... 还有{len(example['differences']) - 5}个差异")
        
        result['differences'].extend(inconsistent_episodes)
    
    # 判断整体一致性
    all_counts_equal = len(set(result['episode_counts'].values())) == 1
    no_missing_extra = not any(result['missing_episodes'].values()) and not any(result['extra_episodes'].values())
    all_consistent = all(details['consistent'] == details['total_common'] 
                        for details in result['consistency_details'].values())
    
    result['all_consistent'] = all_counts_equal and no_missing_extra and all_consistent
    
    if result['all_consistent']:
        print(f"\n🎉 {folder_name} 文件夹数据完全一致!")
    else:
        print(f"\n⚠️  {folder_name} 文件夹存在数据不一致问题")
    
    return result

def main():
    """主函数：验证所有六个数据集文件夹"""
    # 根据实际文件夹名称
    folders = [
        'Environment State',
        'Precise Movement', 
        'Space Structure',
        'Spatial Relationship',
        'Vertical Perception',
        'Viewpoint Shifting'
    ]
    
    print("🚀 开始验证NavSpace数据集的完整性和一致性...")
    print("📋 检查项目：episode总数、ID对应关系、instruction、坐标、目标点等")
    
    all_results = {}
    overall_consistent = True
    
    for folder in folders:
        try:
            result = validate_folder_integrity(folder)
            all_results[folder] = result
            if not result['all_consistent']:
                overall_consistent = False
        except Exception as e:
            print(f"❌ 验证文件夹 {folder} 时出错: {e}")
            overall_consistent = False
    
    # 生成总结报告
    print(f"\n{'='*100}")
    print("📋 数据集验证总结报告")
    print(f"{'='*100}")
    
    total_episodes = 0
    for folder, result in all_results.items():
        status = "✅ 完全一致" if result['all_consistent'] else "❌ 存在问题"
        vln_count = result['episode_counts'].get('vln', 0)
        action_count = result['episode_counts'].get('action', 0)
        tokens_count = result['episode_counts'].get('tokens', 0)
        
        print(f"{folder:20} : {status}")
        print(f"{'':22} VLN: {vln_count:3d} | Action: {action_count:3d} | Tokens: {tokens_count:3d}")
        
        if not result['all_consistent']:
            for file_type, details in result['consistency_details'].items():
                if details['inconsistent'] > 0:
                    print(f"{'':22} └─ {file_type}: {details['inconsistent']}个不一致 ({details['consistency_rate']:.1f}%一致率)")
        
        total_episodes += vln_count
        print()
    
    print(f"总episode数量: {total_episodes}")
    
    if overall_consistent:
        print(f"\n🎉 验证完成！所有数据集文件夹的数据都完全一致！")
    else:
        print(f"\n⚠️  验证完成！发现部分数据集存在不一致问题，请查看上述详细信息。")
        print(f"💡 建议：以vln.json作为基准，修正其他文件中的不一致数据。")

if __name__ == "__main__":
    main()
