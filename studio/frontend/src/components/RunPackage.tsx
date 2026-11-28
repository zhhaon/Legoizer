import { tr, useLocale } from '@/i18n';
import { useStudio } from '@/store/graphStore';
export function RunPackage() {
  useLocale();
  return <section className="min-w-0 shrink-0 border-b border-line p-2 text-xs">
    <button className="tool-btn outline" onClick={() => useStudio.setState({ runGuideOpen: true })}>{tr('guide.entry')}</button>
    <p className="mt-1 text-muted">{tr('guide.entryHelp')}</p>
  </section>;
}
