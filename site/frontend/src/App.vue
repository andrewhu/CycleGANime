<template>
  <div id="app">
    <div id="top"></div>
    <div id="left"></div>
    <div id="right"></div>
    <div id="bottom"></div>
    <router-view/>
  </div>
</template>

<style>
  @font-face {
    font-family: "Lovely Crafter";
    src: url('/fonts/Lovely Crafter.woff2') format('woff2'),
        url('/fonts/Lovely Crafter.woff') format('woff');
  }
  @font-face {
    font-family: "Ubuntu Mono";
    src: url('/fonts/UbuntuMono-Regular.woff2') format('woff2'),
        url('/fonts/UbuntuMono-Regular.woff') format('woff'),
        url('/fonts/UbuntuMono-Regular.ttf') format('truetype');
  }
  @font-face {
    font-family: "Proxima Nova";
    src: url('/fonts/ProximaNovaSoft-Regular.woff2') format('woff2'),
         url('/fonts/ProximaNovaSoft-Regular.woff') format('woff'),
         url('/fonts/ProximaNovaSoft-Regular.ttf') format('truetype');
  }


  /* CSS RESET */
  html, body, div, span, applet, object, iframe,
  h1, h2, h3, h4, h5, h6, p, blockquote, pre,
  a, abbr, acronym, address, big, cite, code,
  del, dfn, em, img, ins, kbd, q, s, samp,
  small, strike, strong, sub, sup, tt, var,
  b, u, i, center,
  dl, dt, dd, ol, ul, li,
  fieldset, form, label, legend,
  table, caption, tbody, tfoot, thead, tr, th, td,
  article, aside, canvas, details, embed,
  figure, figcaption, footer, header, hgroup,
  menu, nav, output, ruby, section, summary,
  time, mark, audio, video {
    margin: 0;
    padding: 0;
    border: 0;
    font-size: 100%;
    font: inherit;
    vertical-align: baseline;
  }

  /* HTML5 display-role reset for older browsers */
  article, aside, details, figcaption, figure,
  footer, header, hgroup, menu, nav, section {
    display: block;
  }

  body {
    line-height: 1;
  }

  ol, ul {
    list-style: none;
  }

  blockquote, q {
    quotes: none;
  }

  blockquote:before, blockquote:after,
  q:before, q:after {
    content: '';
    content: none;
  }

  table {
    border-collapse: collapse;
    border-spacing: 0;
  }

  /* END CSS RESET */

  ::selection {
    background: transparent;
  }

  ::-moz-selection {
    background: transparent;
  }

  /*Page fade in*/
  @-moz-keyframes pageFadeIn {
    0% {
      opacity: 0;
    }
    100% {
      opacity: 1;
    }
  }
  @-ms-keyframes pageFadeIn {
    0% {
      opacity: 0;
    }
    100% {
      opacity: 1;
    }
  }
  @-webkit-keyframes pageFadeIn {
    0% {
      opacity: 0;
    }
    100% {
      opacity: 1;
    }
  }
  @keyframes pageFadeIn {
    0% {
      opacity: 0;
    }
    100% {
      opacity: 1;
    }
  }

  html, body {
    width: 100%;
    height: 100%;
    /*overflow: hidden;*/
    max-width: 100vw;
    max-height: 100vh;
  }

  body {
    background: #f9e0fa;
    font-family: "Lovely Crafter", sans-serif;
    color: #f9276f;
    cursor: default;
    top: 0;
    position: absolute;
    z-index: -2;
    /*overflow: hidden;*/
    /* Page fade in */
    -moz-animation: pageFadeIn 1000ms;
    -ms-animation: pageFadeIn 1000ms;
    -webkit-animation: pageFadeIn 1000ms;
    animation: pageFadeIn 1000ms;
    /*padding-bottom: 100px;*/
  }

  #top, #left, #right, #bottom {
    position: fixed;
    display: block;
    z-index: 999;
    background: #f9276f;
  }
  #top {
    top: 0;
    left: 0;
    right: 0;
    height: 6px;
    width: 100%;
  }
  #left {
    left: 0;
    top: 0;
    bottom: 0;
    width: 6px;
    height: 100%;
  }
  #right {
    right: 0;
    top: 0;
    bottom: 0;
    width: 6px;
    height: 100%;
  }
  #bottom {
    bottom: 0;
    left: 0;
    right: 0;
    height: 6px;
    width: 100%;
  }
  a, a:visited {
    color: #f9276f;
    transition: 150ms ease;
  }

  a:hover {
    color: #fc93b7
  }
</style>