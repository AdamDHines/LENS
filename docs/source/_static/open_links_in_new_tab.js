document.addEventListener("DOMContentLoaded", function () {
  const links = document.querySelectorAll("a.external");
  links.forEach(link => {
    link.setAttribute("target", "_blank");
    link.setAttribute("rel", "noopener noreferrer");
  });
});