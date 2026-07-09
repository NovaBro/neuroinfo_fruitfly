/**
 * Keeps several vtk.js render windows' cameras in lock-step: when the user
 * rotates/zooms/pans one view, the others adopt the same camera pose and
 * re-render. All linked views must share the same world coordinate space (as the
 * FISBe raw/gt/merged volumes of one sample do) for this to read correctly.
 */

interface SyncCamera {
  getPosition(): number[];
  getFocalPoint(): number[];
  getViewUp(): number[];
  setPosition(x: number, y: number, z: number): void;
  setFocalPoint(x: number, y: number, z: number): void;
  setViewUp(x: number, y: number, z: number): void;
  onModified(cb: () => void): { unsubscribe(): void };
}

interface SyncRenderer {
  resetCameraClippingRange(): void;
}

interface SyncRenderWindow {
  render(): void;
}

export interface CameraSyncMember {
  id: string;
  camera: SyncCamera;
  renderer: SyncRenderer;
  renderWindow: SyncRenderWindow;
}

type Pose = {
  position: number[];
  focalPoint: number[];
  viewUp: number[];
};

function poseOf(camera: SyncCamera): Pose {
  return {
    position: camera.getPosition().slice(0, 3),
    focalPoint: camera.getFocalPoint().slice(0, 3),
    viewUp: camera.getViewUp().slice(0, 3),
  };
}

function applyPose(camera: SyncCamera, pose: Pose): void {
  camera.setPosition(pose.position[0], pose.position[1], pose.position[2]);
  camera.setFocalPoint(
    pose.focalPoint[0],
    pose.focalPoint[1],
    pose.focalPoint[2],
  );
  camera.setViewUp(pose.viewUp[0], pose.viewUp[1], pose.viewUp[2]);
}

export class CameraSync {
  private members = new Map<string, CameraSyncMember>();
  private master: Pose | null = null;
  /** >0 while applying poses programmatically, so echoed events are ignored. */
  private suppressDepth = 0;
  private rafScheduled = false;
  private pendingSourceId: string | null = null;
  private enabled: boolean;
  /** Preferred source when (re)linking — the main viewer. */
  private primaryId: string;

  constructor(primaryId = "main", enabled = true) {
    this.primaryId = primaryId;
    this.enabled = enabled;
  }

  /** Register a view; returns an unregister fn (unsubscribes + removes it). */
  register(member: CameraSyncMember): () => void {
    this.members.set(member.id, member);
    const sub = member.camera.onModified(() => this.handleModified(member.id));
    return () => {
      sub.unsubscribe();
      this.members.delete(member.id);
    };
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (!enabled) return;
    // On re-link, snap everyone to the primary view's current camera.
    const primary =
      this.members.get(this.primaryId) ?? this.firstMember();
    if (!primary) return;
    this.master = poseOf(primary.camera);
    this.applyToOthers(primary.id);
  }

  private firstMember(): CameraSyncMember | undefined {
    return this.members.values().next().value;
  }

  private handleModified(sourceId: string): void {
    if (!this.enabled || this.suppressDepth > 0) return;
    this.pendingSourceId = sourceId;
    if (this.rafScheduled) return;
    this.rafScheduled = true;
    requestAnimationFrame(() => {
      this.rafScheduled = false;
      const sourceId = this.pendingSourceId;
      this.pendingSourceId = null;
      if (sourceId == null || !this.enabled) return;
      const source = this.members.get(sourceId);
      if (!source) return;
      this.master = poseOf(source.camera);
      this.applyToOthers(sourceId);
    });
  }

  private applyToOthers(sourceId: string): void {
    if (!this.master) return;
    this.suppressDepth += 1;
    try {
      for (const [id, member] of this.members) {
        if (id === sourceId) continue;
        applyPose(member.camera, this.master);
        member.renderer.resetCameraClippingRange();
        member.renderWindow.render();
      }
    } finally {
      this.suppressDepth -= 1;
    }
  }
}
