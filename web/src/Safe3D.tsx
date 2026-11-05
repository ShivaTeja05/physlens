import { Component, type ReactNode } from 'react'

// Some browsers cannot draw 3D at all: Chrome with graphics acceleration turned off, or a GPU it has blocked,
// reports "Error creating WebGL context". Without this guard that error took the whole page down to a blank screen.

let webgl: boolean | undefined
function webglAvailable(): boolean {
  if (webgl === undefined) {
    try {
      const c = document.createElement('canvas')
      const gl = c.getContext('webgl2') ?? c.getContext('webgl')
      webgl = !!gl
      gl?.getExtension('WEBGL_lose_context')?.loseContext()   // give the probe's context straight back
    } catch {
      webgl = false
    }
  }
  return webgl
}

/** Draws its 3D children where WebGL works; elsewhere, and if drawing fails, says how to turn it on. */
export default class Safe3D extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: !webglAvailable() }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  render() {
    if (!this.state.failed) return this.props.children
    return (
      <div className="no-3d" role="alert">
        <div className="no-3d-title">This browser can’t draw 3D right now</div>
        <p>
          WebGL is turned off. In Chrome, open Settings › System, switch on “Use graphics acceleration when
          available”, relaunch Chrome, then reload this page. chrome://gpu shows whether WebGL is on.
        </p>
      </div>
    )
  }
}
