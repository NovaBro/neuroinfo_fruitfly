import { ReactNode } from "react";
import "./Layout.css";

interface LayoutProps {
  header: ReactNode;
  sidebar: ReactNode;
  main: ReactNode;
  rightPanel?: ReactNode;
}

export function Layout({ header, sidebar, main, rightPanel }: LayoutProps) {
  return (
    <div className="layout">
      <header className="layout__header">{header}</header>
      <div className="layout__body">
        <aside className="layout__sidebar">{sidebar}</aside>
        <main className="layout__main">{main}</main>
        {rightPanel && (
          <aside className="layout__rightpanel">{rightPanel}</aside>
        )}
      </div>
    </div>
  );
}
