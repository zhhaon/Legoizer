import { useLocale } from '@/i18n';
import { rt } from '@/runtime/messages';
import { localMode } from '@/runtime/mode';
import { useStudio } from '@/store/graphStore';

export function RuntimeNotice() {
  useLocale();
  return <button className="fixed bottom-1 right-2 z-50 rounded border border-line bg-surface px-2 py-1 text-xs shadow" onClick={() => useStudio.setState({ bottomTab: 'yaml', bottomOpen: true, welcomeVisible: false })}>{localMode ? rt('local') : rt('server')} · {rt('export')}</button>;
}
