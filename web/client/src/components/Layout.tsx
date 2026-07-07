import { ReactNode, useState } from "react";
import "./Layout.css";

interface LayoutProps {
  header: ReactNode;
  sidebar: ReactNode;
  main: ReactNode;
  rightPanel?: ReactNode;
}

export function Layout({ header, sidebar, main, rightPanel }: LayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [rightOpen, setRightOpen] = useState(true);

  return (
    <div className="layout">
      <header className="layout__header">{header}</header>
      <div className="layout__body">
        {sidebarOpen ? (
          <aside className="layout__sidebar">
            <div className="layout__panel-bar">
              <span className="layout__panel-title">Samples</span>
              <button
                type="button"
                className="layout__collapse-btn"
                onClick={() => setSidebarOpen(false)}
                title="Hide sample list"
                aria-label="Hide sample list"
              >
                ‹
              </button>
            </div>
            <div className="layout__panel-body">{sidebar}</div>
          </aside>
        ) : (
          <button
            type="button"
            className="layout__rail layout__rail--left"
            onClick={() => setSidebarOpen(true)}
            title="Show sample list"
            aria-label="Show sample list"
          >
            <span className="layout__rail-icon">›</span>
            <span className="layout__rail-text">Samples</span>
          </button>
        )}

        <main className="layout__main">{main}</main>

        {rightPanel &&
          (rightOpen ? (
            <aside className="layout__rightpanel">
              <div className="layout__panel-bar">
                <button
                  type="button"
                  className="layout__collapse-btn"
                  onClick={() => setRightOpen(false)}
                  title="Hide metrics"
                  aria-label="Hide metrics"
                >
                  ›
                </button>
                <span className="layout__panel-title">Metrics</span>
              </div>
              <div className="layout__panel-body">{rightPanel}</div>
            </aside>
          ) : (
            <button
              type="button"
              className="layout__rail layout__rail--right"
              onClick={() => setRightOpen(true)}
              title="Show metrics"
              aria-label="Show metrics"
            >
              <span className="layout__rail-icon">‹</span>
              <span className="layout__rail-text">Metrics</span>
            </button>
          ))}
      </div>
    </div>
  );
}
